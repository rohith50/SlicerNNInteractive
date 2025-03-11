import sys
import time
import json
import os
import warnings

from tqdm import tqdm
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from numpysocket import NumpySocket
import requests

import numpy as np
import io
from scipy import ndimage
import torch
import torch.nn as nn
from torch.nn import functional as F
# from tiny_vit_sam import TinyViT
from PIL import Image
import SimpleITK as sitk

from fastapi import FastAPI, Request, Response, UploadFile, File, Form
import gzip

# debug
import matplotlib.pyplot as plt

import hashlib
import gc
import traceback
import xxhash

from sam2_numpy_predictor import build_sam2_numpy_predictor

# 0. freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


image = None
name = None
progress_done = False
app = FastAPI()


def use_torch_bfloat16():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class SAM2:
    def __init__(self,
                 sam2_checkpoint="segment-anything-2/checkpoints/sam2_hiera_large.pt",
                 model_cfg="sam2_hiera_l.yaml"
                 ):
        use_torch_bfloat16()
        
        self.vol = None
        self.annotated_points = None
        self.inference_state = None
        self.segmented_mask = None
        self.min_slice_propagate = None
        self.start_slice_propagate = None
        self.max_slice_propagate = None
        self.all_prompted_slices = set()
        
        self.predictor = build_sam2_numpy_predictor(model_cfg, sam2_checkpoint)
        
    def reset_sam(self):
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        
        if self.vol is None:
            self.segmented_mask = None
        else:
            self.make_empty_mask()
        self.annotated_points = {}
        self.all_prompted_slices = set()
        self.min_slice_propagate = None
        self.start_slice_propagate = None
        self.max_slice_propagate = None
    
    def set_image(self, image):
        self.vol = image
        self.inference_state = self.predictor.init_state(volume=self.vol, offload_video_to_cpu=len(self.vol) > 230)

        self.reset_sam()
    
    def get_points_labels(self, z):
        if z not in self.annotated_points:
            return None
        plist = self.annotated_points[z]
        points = np.array(
            [
                [p['x'], p['y']] 
                for p in plist
            ],  dtype=np.float32
        )
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([p['c'] for p in plist], np.int32)
        
        return points, labels
        
    def make_empty_mask(self):
        self.segmented_mask = np.zeros(self.vol.shape, dtype=np.uint8)
    
    def run_sam2_one_slice_points(self, z):
        ann_obj_id = 1
        
        # x and y points
        points, labels = self.get_points_labels(z)
            
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=self.inference_state,
            frame_idx=z,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
            
        cur_slice_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        
        if self.segmented_mask is None:
            self.make_empty_mask()
        
        self.segmented_mask[z] = cur_slice_mask
        
        return cur_slice_mask
    
    def add_point_to_annotated_points_dict(self, point_xyzc):
        x, y, z, c = point_xyzc
        if z not in self.annotated_points:
            self.annotated_points[z] = []

        if self.start_slice_propagate is None:
            self.start_slice_propagate = z  # This makes the first slice in which is annotated, 
                                              # the slice from which SAM2 propagates.
                    
        self.annotated_points[z].append({'x': x, 'y': y, 'c': c})
    
    def add_new_mask_prompt(self, z, mask):
        if self.segmented_mask is None:
            self.make_empty_mask()

        self.segmented_mask[z] = mask
        ann_obj_id = 1

        if self.start_slice_propagate is None:
            self.start_slice_propagate = z
            print(f'Set self.start_slice_propagate to {self.start_slice_propagate}')

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=z,
            obj_id=ann_obj_id,
            mask=mask
        )
            
        cur_slice_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        
        if self.segmented_mask is None:
            self.make_empty_mask()
        
        self.segmented_mask[z] = cur_slice_mask
        
        self.all_prompted_slices.add(z)

        self.update_propagation_slices_auto()
        
        return cur_slice_mask

    def add_points(self, zyx, positive_click=True):
        print('zyx:', zyx)
        z, y, x = zyx
        # x, y = int(event.xdata), int(event.ydata)
        if positive_click:  # Left-click
            # TODO: add left click right click
            self.add_point_to_annotated_points_dict([x, y, z, 1])
        else:  # Right-click
            self.add_point_to_annotated_points_dict([x, y, z, 0])
        
        points, labels = self.get_points_labels(z)
        
        self.all_prompted_slices.add(z)

        self.run_sam2_one_slice_points(z)
        
        self.update_propagation_slices_auto()

    def update_propagation_slices_auto(self, override=True):
        # print('doing update_propagation_slices_auto')
        if self.min_slice_propagate is None or override:
            self.min_slice_propagate = min(self.all_prompted_slices)
        if self.max_slice_propagate is None or override:
            self.max_slice_propagate = max(self.all_prompted_slices)

    def propagate_sam2_generator(self, use_custom_processing_order=True):
        self.update_propagation_slices_auto(override=False)

        min_slice_propagate, max_slice_propagate = self.min_slice_propagate, self.max_slice_propagate
        if min_slice_propagate is not None and max_slice_propagate is not None:
            if min_slice_propagate > max_slice_propagate:
                min_slice_propagate, max_slice_propagate = max_slice_propagate, min_slice_propagate
            total_slices = max_slice_propagate - min_slice_propagate + 1
        else:
            total_slices = len(self.vol)
        
        processed_slices = 0

        if not use_custom_processing_order:
            for reverse in [False, True]:
                max_frame_num_to_track = None
                if not reverse and max_slice_propagate is not None:
                    max_frame_num_to_track = max_slice_propagate - self.start_slice_propagate + 1
                if reverse and min_slice_propagate is not None:
                    max_frame_num_to_track = self.start_slice_propagate - min_slice_propagate + 1

                if not reverse:
                    processing_order = np.arange(self.start_slice_propagate, self.start_slice_propagate + max_frame_num_to_track)
                else:
                    processing_order = np.arange(self.start_slice_propagate + max_frame_num_to_track, self.start_slice_propagate)

                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        self.inference_state, 
                        reverse=reverse, 
                        start_frame_idx=self.start_slice_propagate,
                        max_frame_num_to_track=max_frame_num_to_track,
                        processing_order=processing_order
                    ):
                    self.segmented_mask[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
                    
                    if self.segmented_mask[out_frame_idx].sum() == 0:
                        break

                    processed_slices += 1
                    progress = min(99, int((processed_slices / total_slices) * 100))
                    yield self.segmented_mask.copy(), progress
        else:
            slices_to_propagate = np.arange(min_slice_propagate, max_slice_propagate + 1)
            prompted_slices_np = np.array(list(self.all_prompted_slices))
            all_dists_slices_to_propagate = []
            for slc in slices_to_propagate:
                dst = np.min(np.abs(slc - prompted_slices_np))
                all_dists_slices_to_propagate.append(dst)
            
            processing_order = slices_to_propagate[np.argsort(all_dists_slices_to_propagate)]
            processing_order = [int(a) for a in processing_order]
            print('\n' * 2)
            print('processing_order:', processing_order)
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        self.inference_state, 
                        reverse=True, 
                        start_frame_idx=self.start_slice_propagate,
                        max_frame_num_to_track=None,
                        processing_order=processing_order
                    ):
                    self.segmented_mask[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
                    
                    if self.segmented_mask[out_frame_idx].sum() == 0:
                        break

                    processed_slices += 1
                    progress = min(99, int((processed_slices / total_slices) * 100))
                    yield self.segmented_mask.copy(), progress

    def remove_prompt_from_slice(self, z):
        ann_obj_id = 1  # Assuming single object with ID 1
        # Call the new method in the predictor
        self.predictor.remove_prompt_from_frame(self.inference_state, frame_idx=z, obj_id=ann_obj_id)
        # Reset the segmentation mask for that slice
        self.segmented_mask[z] = 0
        # Remove annotated points for the slice if they exist
        if z in self.annotated_points:
            del self.annotated_points[z]


sam2 = SAM2()


# def calculate_md5_array(image_data):
#     """Calculate the MD5 checksum of a NumPy array."""
#     md5_hash = hashlib.md5()
#     md5_hash.update(image_data.tobytes())
#     return md5_hash.hexdigest()

def calculate_md5_array(image_data, xx=False):
    """Calculate the MD5 checksum of a NumPy array."""    
    if xx:
        xh = xxhash.xxh64()
        xh.update(image_data.tobytes())
        
        out_hash = xh.hexdigest()
    else:
        md5_hash = hashlib.md5()
        md5_hash.update(image_data.tobytes())
        out_hash = md5_hash.hexdigest()
    
    return out_hash


CACHE_DIR = "/home/user/tmp/sam2_images"
if not os.path.exists(CACHE_DIR):
    CACHE_DIR = ".tmp"


def cache_image(image_data, cache_dir=CACHE_DIR, max_cache=10):
    """Cache image data on disk with MD5 hash as filename."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    md5_hash = calculate_md5_array(image_data)
    file_path = os.path.join(cache_dir, f"{md5_hash}.npy")
    
    if os.path.exists(file_path):
        print("Image already cached.")
        return file_path
    
    np.save(file_path, image_data)
    
    cached_files = sorted(
        [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)], key=os.path.getctime
    )
    
    if len(cached_files) > max_cache:
        os.remove(cached_files[0])
    
    return file_path



from typing import Tuple
#
# 2) check_if_in_cache: returns (True, np_array) if a file named {md5_hash}.npy is in CACHE_DIR
#                       else (False, None)
#
def check_if_in_cache(md5_hash: str) -> Tuple[bool, np.ndarray]:
    file_path = os.path.join(CACHE_DIR, f"{md5_hash}.npy")
    return os.path.exists(file_path)

#
# 3) cache_image: saves the given arr to a local file named {md5_hash}.npy
#
def cache_image(md5_hash: str, arr: np.ndarray):
    file_path = os.path.join(CACHE_DIR, f"{md5_hash}.npy")
    np.save(file_path, arr)

#
# 4) A small endpoint to check if the hash is already cached
#
@app.post("/check_cache")
async def check_cache(md5_hash: str = Form(...)):
    in_cache = check_if_in_cache(md5_hash)
    return {"in_cache": in_cache}

#
# 5) /setImage: only called if not in cache; receives the .npy.gz file plus form fields
#
@app.post("/upload_image")
async def set_image(
    md5_hash: str = Form(...),
    spacings: str = Form(...),
    name_: str = Form(...),
    file: UploadFile = File(None),
):
    global image
    global progress_done
    global name

    # Convert the JSON string "spacings" to a Python list, if needed:
    spacing_list = json.loads(spacings)
    
    if check_if_in_cache(md5_hash):
        file_path = os.path.join(CACHE_DIR, f"{md5_hash}.npy")
        arr = np.load(file_path)
    else:
        # Read the uploaded file bytes, then gzip-decompress
        file_bytes = await file.read()
        decompressed = gzip.decompress(file_bytes)

        # Load the numpy array from the decompressed data
        arr = np.load(io.BytesIO(decompressed))

    # Example: create a SimpleITK image and store globally
    image = sitk.GetImageFromArray(arr)

    # If you have a "sam2" global or other logic, do:
    sam2.set_image(arr)

    # Cache on disk for future re-use
    cache_image(md5_hash, arr)

    progress_done = True
    name = name_  # track the name if needed
    
    return {"status": "ok"}

class ImageParams(BaseModel):
    spacings: list[float]
    name: str
    md5_hash: str

@app.post("/get_progress")
async def get_progress():
    try:
        image_shape_0 = image.shape[0]
    except AttributeError:
        image_shape_0 = -1
    # return json.dumps({'layers': image_shape_0, 'generated_embeds': len(embeddings)})
    embeds_temp_result = image_shape_0 if progress_done else -1
    return json.dumps({'layers': image_shape_0, 'generated_embeds': embeds_temp_result})

@app.post("/get_server_state")
async def get_server_state():
    return json.dumps({'ready': True})

@app.post("/get_image_name")
async def get_image_loaded_state():
    if image is None:
        return json.dumps({'name': None})
    else:
        return json.dumps({'name': name})

@app.post("/get_numpy_state")
# check image is completely sent
async def get_numpy_state():
    if progress_done:
        return json.dumps({'ready': True}) # 'shape': image_block.shape
    else:
        return json.dumps({'ready': False}) # 'shape': None
    
@app.post("/print_hello")
async def printHello():
    print('Hello world')
    return ('Hello world returned')

@app.post("/reset_sam")
async def reset_sam():
    sam2.reset_sam()

class SliceParams(BaseModel):
    z: int
    
# Add new endpoints
@app.post("/start_propagation")
async def start_propagation(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_propagation_in_background)
    return {"status": "Propagation started"}

import threading
partial_results = {}
partial_results_lock = threading.Lock()

# Add a global flag to signal propagation termination
terminate_propagation_flag = threading.Event()

@app.post("/terminate_propagation")
async def terminate_propagation():
    terminate_propagation_flag.set()
    return {"status": "Propagation termination requested"}

def run_propagation_in_background():
    partial_results.clear()
    terminate_propagation_flag.clear()
    
    for partial_result, progress in sam2.propagate_sam2_generator():
        if terminate_propagation_flag.is_set():
            break  # Stop propagation if termination is requested
        
        with partial_results_lock:
            partial_results['latest'] = partial_result.copy()
            partial_results['progress'] = progress  # Store progress
    
    with partial_results_lock:
        partial_results['finished'] = True


@app.post("/get_partial_result")
async def get_partial_result():
    with partial_results_lock:
        if 'latest' in partial_results:
            segmentation_binary_data = segmentation_binary(partial_results['latest'], compress=True)
            response = Response(content=segmentation_binary_data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})
            
            response.headers['Propagation-Status'] = 'Finished' if 'finished' in partial_results and partial_results['finished'] else 'InProgress'
            response.headers['Propagation-Progress'] = str(partial_results.get('progress', 0))  # Add progress header
            
            return response
        else:
            return Response(content=b'', media_type="application/octet-stream")
        
@app.post("/set_min_slice_propagate")
async def set_min_slice_propagate(params: SliceParams):
    sam2.min_slice_propagate = params.z
        
@app.post("/set_start_slice_propagate")
async def set_start_slice_propagate(params: SliceParams):
    sam2.start_slice_propagate = params.z
    
@app.post("/set_max_slice_propagate")
async def set_max_slice_propagate(params: SliceParams):
    sam2.max_slice_propagate = params.z

@app.post("/get_segmentation_hash")
def get_segmentation_hash():
    if sam2.segmented_mask is None:
        return {"hash": ""}  # Return a dict, not json.dumps(...)
    seg_array = sam2.segmented_mask
    md5_hash = calculate_md5_array(seg_array.astype(bool))
    print('np.sum(seg_array.astype(bool)):', np.sum(seg_array.astype(bool)))
    print('md5_hash:', md5_hash)
    print('seg_array.shape:', seg_array.shape)
    return {"hash": md5_hash}


class InferenceParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool

@app.post("/infer")
async def infer(params: InferenceParams):
    t = time.time()
    if image is None:
        warnings.warn('There is no image in the server. Be sure to send it before')
        return []
    
    positive_click = params.positive_click
    
    xyz = params.voxel_coord
    zyx = xyz[::-1]
    
    _ = sam2.add_points(zyx, positive_click=positive_click)
    seg_result = sam2.segmented_mask
    
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)
    print(f'Server whole infer function time: {time.time() - t}')
    
    # Return as binary data with appropriate content type
    return Response(content=segmentation_binary_data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})

class MaskPromptParams(BaseModel):
    z: int

@app.post("/mask_prompt")
async def mask_prompt(request: Request, z: int = Form(...), file: UploadFile = File(...)):
    print('doing mask prompt!')
    # Read the binary data
    binary_data = await file.read()

    print('z:', z)

    vol_shape = None
    if z != -1:  # z == -1 means that a whole volume is sent and all non-empty slices are a mask prompt
        vol_shape = sam2.vol.shape[1:]

    mask_prompt = unpack_binary_segmentation(gzip.decompress(binary_data),
                                             vol_shape=vol_shape)

    if z != -1:
        _ = sam2.add_new_mask_prompt(z, mask_prompt)
    else:  # z == -1"
        for zi in range(sam2.vol.shape[0]):
            if mask_prompt[zi].sum() > 0:
                _ = sam2.add_new_mask_prompt(zi, mask_prompt[zi])
    
    if sam2.segmented_mask is None:
        sam2.make_empty_mask()
            
    segmentation_binary_data = segmentation_binary(sam2.segmented_mask)
    
    # Return a response (you can send a confirmation or process and return more binary data)
    return Response(content=segmentation_binary_data, media_type="application/octet-stream")


class SliceParams(BaseModel):
    zs: list[int]

@app.post("/remove_prompt")
def remove_prompt(params: SliceParams):
    for z in params.zs:
        sam2.remove_prompt_from_slice(z)
    seg_result = sam2.segmented_mask
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)

    print('Sending remove prompt response. segmentation_binary_data size:', len(segmentation_binary_data))
    
    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"}
    )

    
def unpack_binary_segmentation(binary_data, vol_shape=None):
    """
    Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type).
    
    Parameters:
        binary_data (bytes): The packed binary segmentation data.
    
    Returns:
        np.ndarray: The unpacked 3D boolean numpy array.
    """
    # Get the shape of the original volume (same as image_data shape)
    vol_shape = sam2.vol.shape if vol_shape is None else vol_shape
    
    # Calculate the total number of bits (voxels)
    total_voxels = np.prod(vol_shape)
    
    # Unpack the binary data (convert from bytes to bits)
    unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
    
    # Trim any extra bits (in case the bit length is not perfectly divisible)
    unpacked_bits = unpacked_bits[:total_voxels]
    
    # Reshape into the original volume shape
    segmentation_mask = unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
    
    return segmentation_mask

def segmentation_binary(seg_in, compress=False):
    # Assuming seg_in is the boolean segmentation (True for segmented, False for not)
    seg_result = seg_in.astype(bool)  # Convert to bool type if not already
    
    # Pack the boolean array into bytes (1 bit per voxel)
    packed_segmentation = np.packbits(seg_result, axis=None)  # Pack into 1D byte array

    packed_segmentation = packed_segmentation.tobytes()

    if compress:
        # Compress the binary data with gzip
        packed_segmentation = gzip.compress(packed_segmentation)
    
    return packed_segmentation # Convert to bytes for transmission

# Add this new endpoint
@app.post("/get_propagation_slices")
async def get_propagation_slices():
    return json.dumps({
        'min_slice': sam2.min_slice_propagate,
        'start_slice': sam2.start_slice_propagate,
        'max_slice': sam2.max_slice_propagate
    })

if __name__ == "__main__":
    uvicorn.run("server_sam2:app", host="0.0.0.0", port=1526)
