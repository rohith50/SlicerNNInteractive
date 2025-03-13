import logging

from collections import OrderedDict

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tqdm import tqdm

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points

import torch.nn.functional as F

import matplotlib.pyplot as plt


def load_numpy_frames(
    volume,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    num_frames = len(volume)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    images = torch.from_numpy(volume)
    if images.ndim == 3:
        # Assuming (D, H, W), to which an RGB dimension needs to be added to dim 1
        images = torch.stack([images] * 3, 1)
    images = images.to(torch.float32)
    
    images = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False)
    images = (images - images.min()) / (images.max() - images.min())
    
    video_height = volume.shape[1]
    video_width = volume.shape[2]
    
    if not offload_video_to_cpu:
        if type(offload_video_to_cpu) is list:
            images_out = []
            for slice_idx, image in enumerate(images):
                device = "cpu" if slice_idx in offload_video_to_cpu else "cuda"
                images_out.append(
                    (image.to(device) - img_mean.to(device)) / img_std.to(device)
                )
        elif type(offload_video_to_cpu) is bool:
            images = images.cuda()
            img_mean = img_mean.cuda()
            img_std = img_std.cuda()
            
            images -= img_mean
            images /= img_std
        else:
            raise TypeError("offload_video_to_cpu should be either list or bool", type(offload_video_to_cpu))
    else:
        # normalize by mean and std
        images -= img_mean
        images /= img_std
    return images, video_height, video_width

class SAM2NumpyPredictor(SAM2VideoPredictor):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    @torch.inference_mode()
    def update_images_with_new_offload(
        self, 
        volume,
        inference_state, 
        offload_video_to_cpu=False
    ):
        images, _, _ = load_numpy_frames(
            volume=volume,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu
        )
        
        inference_state["images"] = images
        
        return inference_state
        
    
    @torch.inference_mode()
    def init_state(
        self,
        volume,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize a inference state."""
        images, video_height, video_width = load_numpy_frames(
            volume=volume,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state
    
    def remove_prompt_from_frame(self, inference_state, frame_idx, obj_id):
        """Remove prompts and outputs associated with a given frame and object ID."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Remove point inputs for the frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        # Remove mask inputs for the frame
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        # Remove temporary outputs
        inference_state["temp_output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        inference_state["temp_output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
        # Remove outputs from per-object output dicts
        inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
        # Remove outputs from the overall output dict
        inference_state["output_dict"]["cond_frame_outputs"].pop(frame_idx, None)
        inference_state["output_dict"]["non_cond_frame_outputs"].pop(frame_idx, None)
        # Remove frame index from consolidated_frame_inds
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].discard(frame_idx)
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(frame_idx)
        # Remove frame from frames_already_tracked
        inference_state["frames_already_tracked"].pop(frame_idx, None)
        # Optionally, remove cached features for the frame
        # inference_state["cached_features"].pop(frame_idx, None)
        
    
    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        processing_order=None,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
            
        if processing_order is None:
            if reverse:
                end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
                if start_frame_idx > 0:
                    processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
                else:
                    processing_order = []  # skip reverse tracking if starting from frame 0
            else:
                end_frame_idx = min(
                    start_frame_idx + max_frame_num_to_track, num_frames - 1
                )
                processing_order = range(start_frame_idx, end_frame_idx + 1)

        prev_frame_idx = -1
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if processing_order is not None:
                reverse = prev_frame_idx > frame_idx
                prev_frame_idx = frame_idx
            
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks


def build_sam2_numpy_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2_numpy_predictor.SAM2NumpyPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint successfully")
