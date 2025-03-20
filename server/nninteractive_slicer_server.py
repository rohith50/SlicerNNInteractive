import time
import json
import warnings

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

import numpy as np
import io


import os
import torch
import SimpleITK as sitk
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


from fastapi import FastAPI, Request, Response, UploadFile, File, Form
import gzip

import hashlib
import xxhash


app = FastAPI()

# --- Download Trained Model Weights (~400MB) ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "/opt/server/temp"  # Specify the download directory


class PromptManager:
    def __init__(self):
        self.img = None
        self.target_tensor = None
        
        self.download_weights()
        self.session = self.make_session()
    
    def download_weights(self):
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR
        )
    
    def make_session(self):
        session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=True,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        
        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session
    
    def set_image(self, input_image):
        self.session.reset_interactions()
        
        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)
        
        print('self.img.shape:', self.img.shape)
        
        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(self.img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)
    
    def set_segment(self, mask):
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(self.img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask)
    
    def add_point_interaction(self, point_coordinates, include_interaction):
        print('self.session.target_buffer.shape:', self.session.target_buffer.shape)
        print('self.img.shape:', self.img.shape)
        
        # point_coordinates is (x, y, z)
        print('point_coordinates:', point_coordinates)
        self.session.add_point_interaction(point_coordinates, include_interaction=include_interaction)
        
        return self.target_tensor.clone().cpu().detach().numpy()
    
    def add_bbox_interaction(self, outer_point_one, outer_point_two, include_interaction):
        # outer_point_one and outer_point_two are (x, y, z) coordinates.
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)
        
        # outer_point_one = [170, 170, 39]
        # outer_point_two = [230, 230, 39]
        
        # Create an array from the two points and compute min and max for each coordinate.
        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)
        
        # Construct the bounding box as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]
        
        print('bbox:', bbox)
        
        # Call the session's bounding box interaction function.
        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)
        
        return self.target_tensor.clone().cpu().detach().numpy()
    
    # def add_bbox_interaction(self, outer_point_one, outer_point_two, include_interaction):
    #     # outer_point_one, outer_point_two are (x, y, z). They represent, for example, the top left and bottom right points.
    #     print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)
        
    #     outer_point_one = [170, 170, 39]
    #     outer_point_two = [230, 230, 39]
        
    #     # Unpack the coordinates
    #     x1, y1, z1 = outer_point_one
    #     x2, y2, z2 = outer_point_two
        
    #     # Define the 2D bounding box in the axial (XY) plane:
    #     # For x and y, take the min and max values.
    #     # For z, choose a single slice by taking the lower z value and defining the interval as [z, z+1].
    #     bbox_coordinates = [
    #         [np.array(int(min(x1, x2))), np.array(int(max(x1, x2)))],  # X: convert to Python ints
    #         [np.array(int(min(y1, y2))), np.array(int(max(y1, y2)))],  # Y: convert to Python ints
    #         [np.array(int(min(z1, z2))), np.array(int(min(z1, z2))) + 1]  # Z: single slice
    #     ]
        
    #     print('bbox_coordinates:', bbox_coordinates)
        
    #     self.session.add_point_interaction(bbox_coordinates[::-1], include_interaction=include_interaction)
        
    #     return self.target_tensor.clone().cpu().detach().numpy()


PROMPT_MANAGER = PromptManager()


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


@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(None),
):
    # Read the uploaded file bytes, then gzip-decompress
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)

    # Load the numpy array from the decompressed data
    arr = np.load(io.BytesIO(decompressed))
    PROMPT_MANAGER.set_image(arr)
    
    return {"status": "ok"}

@app.post("/upload_segment")
async def upload_segment(
    file: UploadFile = File(None),
):
    # Read the uploaded file bytes, then gzip-decompress
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)

    # Load the numpy array from the decompressed data
    arr = np.load(io.BytesIO(decompressed))
    # PROMPT_MANAGER.set_target_tensor(arr)
    
    PROMPT_MANAGER.set_segment(arr)
    
    return {"status": "ok"}



# @app.post("/set_mask")
# async def set_mask(file: UploadFile = File(...)):
#     print('doing mask prompt!')
#     # Read the binary data
#     binary_data = await file.read()

#     vol_shape = PROMPT_MANAGER.target_tensor.shape

#     mask_prompt = unpack_binary_segmentation(gzip.decompress(binary_data),
#                                              vol_shape=vol_shape)

#     PROMPT_MANAGER.target_tensor = torch.from_numpy(mask_prompt).astype(torch.uint8)
    
    
@app.post("/print_hello")
async def print_hello():
    print('Hello world')
    return ('Hello world returned')
    

import threading
partial_results = {}
partial_results_lock = threading.Lock()


@app.post("/get_segmentation_hash")
def get_segmentation_hash():
    if PROMPT_MANAGER.target_tensor is None:
        return {"hash": ""}  # Return a dict, not json.dumps(...)
    seg_array = PROMPT_MANAGER.target_tensor
    md5_hash = calculate_md5_array(seg_array.astype(bool))
    print('np.sum(seg_array.astype(bool)):', np.sum(seg_array.astype(bool)))
    print('md5_hash:', md5_hash)
    print('seg_array.shape:', seg_array.shape)
    return {"hash": md5_hash}


class PointParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool


@app.post("/add_point_interaction")
async def add_point_interaction(params: PointParams):
    t = time.time()
    if PROMPT_MANAGER.img is None:
        warnings.warn('There is no image in the server. Be sure to send it before')
        return []
    
    positive_click = params.positive_click
    
    xyz = params.voxel_coord
    print('xyz:', xyz)
    
    # seg_result = PROMPT_MANAGER.add_point_interaction(xyz)
    seg_result = PROMPT_MANAGER.add_point_interaction(xyz, include_interaction=positive_click)
    
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)
    print(f'Server whole infer function time: {time.time() - t}')
    
    # Return as binary data with appropriate content type
    return Response(content=segmentation_binary_data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})


class BBoxParams(BaseModel):
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_click: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    t = time.time()
    if PROMPT_MANAGER.img is None:
        warnings.warn('There is no image in the server. Be sure to send it before')
        return []
    
    
    # seg_result = PROMPT_MANAGER.add_point_interaction(xyz)
    seg_result = PROMPT_MANAGER.add_bbox_interaction(params.outer_point_one, 
                                                     params.outer_point_two,
                                                     include_interaction=params.positive_click)
    
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)
    print(f'Server whole infer function time: {time.time() - t}')
    
    # Return as binary data with appropriate content type
    return Response(content=segmentation_binary_data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})

    
def unpack_binary_segmentation(binary_data, vol_shape):
    """
    Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type).
    
    Parameters:
        binary_data (bytes): The packed binary segmentation data.
    
    Returns:
        np.ndarray: The unpacked 3D boolean numpy array.
    """
    
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


if __name__ == "__main__":
    uvicorn.run("nninteractive_slicer_server:app", host="0.0.0.0", port=1527)
