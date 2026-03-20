import time
import numpy as np
from cpu_mppca_full import mppca_cpu
from gpu_mppca_full import GpuMPPCAFull

sizes = [
    (24, 32, "24³×32"),
    (32, 32, "32³×32"),
    (32, 64, "32³×64"),
]

print(f"{'Volume':<12} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>10} {'Max diff':>12} {'Correct':>10}")
print("-" * 68)

gpu = GpuMPPCAFull()

for spatial, channels, label in sizes:
    rng = np.random.default_rng(42)
    data = (rng.standard_normal((spatial, spatial, spatial, channels)) * 50 + 300).astype(np.float32)

    t0 = time.perf_counter()
    cpu_result = mppca_cpu(data, patch_radius=1)
    cpu_ms = (time.perf_counter() - t0) * 1000

    gpu.fit(data, patch_radius=1)
    t0 = time.perf_counter()
    gpu_result = gpu.fit(data, patch_radius=1)
    gpu_ms = (time.perf_counter() - t0) * 1000

    speedup = cpu_ms / gpu_ms
    max_diff = np.max(np.abs(cpu_result - gpu_result))
    correct = "PASS" if max_diff < 10.0 else "FAIL"

    print(f"{label:<12} {cpu_ms:>10.1f} {gpu_ms:>10.1f} {speedup:>9.1f}x {max_diff:>12.2f} {correct:>10}")
