import numpy as np
from cpu_mppca_full import mppca_cpu
from gpu_mppca_full import GpuMPPCAFull

rng = np.random.default_rng(42)
data = (rng.standard_normal((8, 8, 8, 8)) * 50 + 300).astype(np.float32)

cpu_result = mppca_cpu(data, patch_radius=1)
gpu = GpuMPPCAFull()
gpu_result = gpu.fit(data, patch_radius=1)

print("Shape match:", cpu_result.shape == gpu_result.shape)
print("Max diff:", np.max(np.abs(cpu_result - gpu_result)))
print("Mean diff:", np.mean(np.abs(cpu_result - gpu_result)))
print("Allclose atol=1.0:", np.allclose(cpu_result, gpu_result, atol=1.0))
print("CPU sample values:", cpu_result[4,4,4,:4])
print("GPU sample values:", gpu_result[4,4,4,:4])
