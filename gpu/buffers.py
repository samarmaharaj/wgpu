# dipy/gpu/buffers.py
import wgpu
import numpy as np
from dipy.gpu.device import DiPyGPUDevice

def _ensure_gpu_aligned(arr: np.ndarray, target_dtype=np.float32) -> np.ndarray:
    """
    Ensures the array is C-contiguous and typed correctly for WGSL.
    Prevents silent memory corruption when uploading sliced arrays.
    """
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr

def upload_array(arr: np.ndarray, usage=None) -> wgpu.GPUBuffer:
    """
    Safely uploads a NumPy array to VRAM. 
    Defaults to STORAGE binding, ready for compute shaders.
    """
    device = DiPyGPUDevice.get()
    if usage is None:
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        
    safe_arr = _ensure_gpu_aligned(arr)
    
    return device.create_buffer_with_data(
        data=safe_arr.tobytes(), 
        usage=usage
    )

def create_empty_buffer(size_bytes: int, usage=None) -> wgpu.GPUBuffer:
    """Allocates uninitialized VRAM for shader outputs."""
    device = DiPyGPUDevice.get()
    if usage is None:
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    
    # WGSL buffers often require 4-byte alignment
    aligned_size = (size_bytes + 3) & ~3 
    
    return device.create_buffer(size=aligned_size, usage=usage)

def readback_array(buffer: wgpu.GPUBuffer, shape: tuple, dtype=np.float32) -> np.ndarray:
    """
    Safely transfers VRAM back to a NumPy array using a staging buffer.
    Abstracts away the command encoder and map/unmap synchronization.
    """
    device = DiPyGPUDevice.get()
    
    # 1. Create a CPU-mappable staging buffer
    staging = device.create_buffer(
        size=buffer.size, 
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )
    
    # 2. Encode the device-to-host copy command
    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, buffer.size)
    device.queue.submit([encoder.finish()])
    
    # 3. Synchronously map, copy to NumPy, and unmap
    staging.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(staging.read_mapped(), dtype=dtype).copy()
    staging.unmap()
    
    # Return the data reshaped to the original DIPY volume dimensions
    return data[:np.prod(shape)].reshape(shape)