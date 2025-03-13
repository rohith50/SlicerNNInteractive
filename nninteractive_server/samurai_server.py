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
DOWNLOAD_DIR = "/home/user/temp"  # Specify the download directory


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
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        
        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session
    
    def set_image(self, input_image):
        # input_image is sitk.Image
        self.img = sitk.GetArrayFromImage(input_image)[None]  # Ensure shape (1, x, y, z)
        
        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(self.img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)
    
    def add_point_interaction(self, point_coordinates):
        # point_coordinates is (x, y, z)
        self.session.add_point_interaction(point_coordinates, include_interaction=True)
        
        return self.target_tensor


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
async def set_image(
    file: UploadFile = File(None),
):
    # Read the uploaded file bytes, then gzip-decompress
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)

    # Load the numpy array from the decompressed data
    arr = np.load(io.BytesIO(decompressed))
    PROMPT_MANAGER.set_image(arr)
    
    return {"status": "ok"}

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


class InferenceParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool


@app.post("/add_point_interaction")
async def add_point_interaction(params: InferenceParams):
    t = time.time()
    if PROMPT_MANAGER.image is None:
        warnings.warn('There is no image in the server. Be sure to send it before')
        return []
    
    positive_click = params.positive_click
    
    xyz = params.voxel_coord
    zyx = xyz[::-1]
    
    seg_result = PROMPT_MANAGER.add_point_interaction(zyx)
    
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)
    print(f'Server whole infer function time: {time.time() - t}')
    
    # Return as binary data with appropriate content type
    return Response(content=segmentation_binary_data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})


@app.post("/set_mask")
async def set_mask(file: UploadFile = File(...)):
    print('doing mask prompt!')
    # Read the binary data
    binary_data = await file.read()

    vol_shape = PROMPT_MANAGER.img.shape[:-1]

    mask_prompt = unpack_binary_segmentation(gzip.decompress(binary_data),
                                             vol_shape=vol_shape)

    PROMPT_MANAGER.target_tensor = torch.from_numpy(mask_prompt).astype(torch.uint8)


    
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
    uvicorn.run("samurai_server:app", host="0.0.0.0", port=1527)
