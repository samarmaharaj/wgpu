"""
Quick-Start Guide for MPPCA Implementations

This file provides copy-paste examples for common usage patterns.
"""

# ============================================================================
# EXAMPLE 1: Basic CPU Usage
# ============================================================================

"""
from cpu_mppca_full import mppca_cpu
import numpy as np

# Generate or load 4D data: (X, Y, Z, channels)
data = np.random.randn(64, 64, 64, 32).astype(np.float32)

# Apply MPPCA denoising (basic)
denoised = mppca_cpu(data)
print(f"Output shape: {denoised.shape}")

# With custom parameters
denoised = mppca_cpu(
    data,
    patch_radius=2,      # Patch size: (2*2+1)³ = 125 samples
    tau_factor=1.5,      # Threshold multiplier (1.0 = default)
    verbose=True         # Print progress
)
"""


# ============================================================================
# EXAMPLE 2: GPU Usage (if available)
# ============================================================================

"""
from gpu_mppca_full import GpuMPPCAFull
import numpy as np

# Generate or load 4D data
data = np.random.randn(64, 64, 64, 32).astype(np.float32)

# One-shot computation
gpu = GpuMPPCAFull()
denoised = gpu.fit(data, patch_radius=2)

# Advanced: preload then compute (for repeated operations)
buf_vol, shape = gpu.preload(data)
denoised = gpu.fit_preloaded(
    buf_vol,
    shape[0], shape[1], shape[2], shape[3],  # X, Y, Z, channels
    patch_radius=2
)
"""


# ============================================================================
# EXAMPLE 3: Comparing CPU vs GPU
# ============================================================================

"""
from cpu_mppca_full import mppca_cpu
from gpu_mppca_full import GpuMPPCAFull
import numpy as np
import time

data = np.random.randn(32, 32, 32, 16).astype(np.float32)

# CPU
t0 = time.time()
result_cpu = mppca_cpu(data, patch_radius=1)
t_cpu = (time.time() - t0) * 1000

# GPU
gpu = GpuMPPCAFull()
t0 = time.time()
result_gpu = gpu.fit(data, patch_radius=1)
t_gpu = (time.time() - t0) * 1000

# Validation
print(f"CPU: {t_cpu:.2f} ms")
print(f"GPU: {t_gpu:.2f} ms")
print(f"Speedup: {t_cpu/t_gpu:.2f}×")
print(f"Max diff: {np.max(np.abs(result_cpu - result_gpu)):.6e}")
print(f"Match: {np.allclose(result_cpu, result_gpu, atol=1e-1)}")
"""


# ============================================================================
# EXAMPLE 4: Parameter Tuning
# ============================================================================

"""
import numpy as np
from cpu_mppca_full import mppca_cpu

data = np.random.randn(64, 64, 64, 32).astype(np.float32)

# Test different patch radii
for r in [1, 2, 3]:
    result = mppca_cpu(data, patch_radius=r, verbose=False)
    print(f"Patch radius {r}: Done")

# Test different tau factors
for tau in [0.8, 1.0, 1.2, 1.5]:
    result = mppca_cpu(data, tau_factor=tau, verbose=False)
    print(f"Tau factor {tau:.1f}: Done")
"""


# ============================================================================
# EXAMPLE 5: Working with Real dMRI Data
# ============================================================================

"""
import nibabel as nib
import numpy as np
from cpu_mppca_full import mppca_cpu

# Load dMRI NIfTI file
img = nib.load("dwi.nii")
data = img.get_fdata().astype(np.float32)  # Shape: (X, Y, Z, N_gradients)

# Denoise
denoised = mppca_cpu(data, patch_radius=2, tau_factor=1.2)

# Save result
result_img = nib.Nifti1Image(denoised, img.affine, img.header)
nib.save(result_img, "dwi_denoised.nii")
"""


# ============================================================================
# EXAMPLE 6: Memory-Efficient Processing (Large Volumes)
# ============================================================================

"""
import numpy as np
from cpu_mppca_full import mppca_cpu

# For very large volumes, process in slices
def denoise_volume_slices(data, slice_axis=2, patch_radius=2):
    X, Y, Z, C = data.shape
    result = np.zeros_like(data)
    
    n_slices = data.shape[slice_axis]
    for i in range(n_slices):
        if slice_axis == 2:
            slice_data = data[:, :, i:i+1, :]
        else:
            raise NotImplementedError("Extend as needed")
        
        # Denoise slice (with appropriate handling of boundaries)
        result[:, :, i:i+1, :] = mppca_cpu(slice_data, patch_radius=patch_radius)
        
        print(f"Processed slice {i+1}/{n_slices}")
    
    return result

# Usage
large_data = np.random.randn(256, 256, 100, 64).astype(np.float32)
denoised = denoise_volume_slices(large_data, patch_radius=2)
"""


# ============================================================================
# EXAMPLE 7: Batch Processing with GPU
# ============================================================================

"""
from gpu_mppca_full import GpuMPPCAFull
import numpy as np

# Process multiple files
gpu = GpuMPPCAFull()

files = ["data1.npy", "data2.npy", "data3.npy"]

for fname in files:
    data = np.load(fname).astype(np.float32)
    denoised = gpu.fit(data, patch_radius=2)
    np.save(fname.replace(".npy", "_denoised.npy"), denoised)
    print(f"Processed {fname}")
"""


# ============================================================================
# EXAMPLE 8: Noise Level Estimation
# ============================================================================

"""
import numpy as np
from cpu_mppca_full import mppca_cpu

def estimate_noise_level(denoised, noisy):
    # MSE between noisy and denoised volumes
    mse = np.mean((noisy - denoised) ** 2)
    noise_std = np.sqrt(mse)
    return noise_std

data_noisy = np.random.randn(64, 64, 64, 32).astype(np.float32) + \
             np.random.randn(64, 64, 64, 32).astype(np.float32) * 50

data_denoised = mppca_cpu(data_noisy, patch_radius=2)

noise_std = estimate_noise_level(data_denoised, data_noisy)
print(f"Estimated noise std: {noise_std:.2f}")
"""


# ============================================================================
# EXAMPLE 9: Performance Monitoring
# ============================================================================

"""
import numpy as np
import time
import psutil
import os
from cpu_mppca_full import mppca_cpu

def monitor_performance(data, patch_radius=2):
    process = psutil.Process(os.getpid())
    
    # Memory before
    mem_before = process.memory_info().rss / (1024**2)  # MB
    
    # Run MPPCA
    t0 = time.time()
    result = mppca_cpu(data, patch_radius=patch_radius, verbose=False)
    elapsed = time.time() - t0
    
    # Memory after
    mem_after = process.memory_info().rss / (1024**2)  # MB
    
    return {
        "time_sec": elapsed,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "mem_peak_mb": mem_after - mem_before,
    }

data = np.random.randn(64, 64, 64, 32).astype(np.float32)
metrics = monitor_performance(data)
print(f"Time: {metrics['time_sec']:.2f}s")
print(f"Memory peak: {metrics['mem_peak_mb']:.1f} MB")
"""


# ============================================================================
# EXAMPLE 10: Automated Parameter Selection
# ============================================================================

"""
import numpy as np
from cpu_mppca_full import mppca_cpu

def auto_param_selection(data):
    '''Automatically set patch_radius and tau_factor based on data shape.'''
    X, Y, Z, d = data.shape
    
    # Rule of thumb: patch radius proportional to sqrt(d)
    patch_radius = max(1, int(np.sqrt(d) / 2))
    
    # Rule of thumb: tau_factor based on oversampling ratio
    patch_size = (2 * patch_radius + 1) ** 3
    oversampling = patch_size / d
    tau_factor = 1.0 + np.sqrt(d / patch_size)
    
    return patch_radius, tau_factor

# Usage
data = np.random.randn(64, 64, 64, 64).astype(np.float32)
r, tau = auto_param_selection(data)
print(f"Recommended: patch_radius={r}, tau_factor={tau:.3f}")

denoised = mppca_cpu(data, patch_radius=r, tau_factor=tau)
"""


# ============================================================================
# EXAMPLE 11: Comparison with Other Denoising Methods
# ============================================================================

"""
import numpy as np
from scipy.ndimage import gaussian_filter
from cpu_mppca_full import mppca_cpu

def compare_denoisers(data):
    # Original
    mse_orig = np.mean(data**2)  # Assuming data is noise
    
    # Gaussian smoothing
    gauss = gaussian_filter(data, sigma=1.0)
    mse_gauss = np.mean((data - gauss) ** 2)
    
    # MPPCA
    mppca = mppca_cpu(data, patch_radius=2)
    mse_mppca = np.mean((data - mppca) ** 2)
    
    print(f"Original variance: {mse_orig:.2e}")
    print(f"Gaussian MSE: {mse_gauss:.2e}")
    print(f"MPPCA MSE: {mse_mppca:.2e}")

data = np.random.randn(32, 32, 32, 16).astype(np.float32)
compare_denoisers(data)
"""


# ============================================================================
# EXAMPLE 12: Runtime Benchmarks
# ============================================================================

"""
# See: benchmark_mppca.py for comprehensive benchmarking

# Quick benchmark
from cpu_mppca_full import mppca_cpu
from gpu_mppca_full import GpuMPPCAFull
import numpy as np
import time

for size in [8, 16, 24]:
    for n_grad in [8, 16, 32]:
        data = np.random.randn(size, size, size, n_grad).astype(np.float32)
        
        # CPU
        t0 = time.time()
        mppca_cpu(data, verbose=False)
        t_cpu = (time.time() - t0) * 1000
        
        # GPU
        try:
            gpu = GpuMPPCAFull()
            t0 = time.time()
            gpu.fit(data)
            t_gpu = (time.time() - t0) * 1000
            speedup = t_cpu / t_gpu
        except:
            t_gpu = None
            speedup = None
        
        msg = f"{size}³×{n_grad} | CPU: {t_cpu:.1f}ms"
        if t_gpu:
            msg += f" | GPU: {t_gpu:.1f}ms | {speedup:.1f}×"
        
        print(msg)
"""


if __name__ == "__main__":
    print(__doc__)
