import time
import numpy as np
from cpu_mppca_full import mppca_cpu
from gpu_mppca_full import GpuMPPCAFull

spatial=32
channels=32
rng=np.random.default_rng(42)
data=(rng.standard_normal((spatial, spatial, spatial, channels))*50+300).astype(np.float32)

gpu=GpuMPPCAFull()

t0=time.perf_counter()
cpu_result=mppca_cpu(data, patch_radius=1)
cpu_ms=(time.perf_counter()-t0)*1000

gpu.fit(data, patch_radius=1)
t0=time.perf_counter()
gpu_result=gpu.fit(data, patch_radius=1)
gpu_ms=(time.perf_counter()-t0)*1000

max_diff=float(np.max(np.abs(cpu_result-gpu_result)))
print(f"cpu_ms={cpu_ms:.1f}")
print(f"gpu_ms={gpu_ms:.1f}")
print(f"max_diff={max_diff:.4f}")
