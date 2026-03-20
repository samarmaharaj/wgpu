import time, numpy as np
from cpu_gibbs_full import gibbs_cpu
from gpu_gibbs_full import GpuGibbsFull

sizes = [
    (32, 1,  "32³×1"),
    (32, 16, "32³×16"),
    (64, 1,  "64³×1"),
    (64, 32, "64³×32"),
    (80, 1,  "80³×1"),
    (80, 32, "80³×32"),
    (128,1,  "128³×1"),
]

print(f"{'Volume':<12} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>10} {'Max diff':>12} {'Correct':>10}")
print("-" * 68)

gpu = GpuGibbsFull()
for spatial, n_grad, label in sizes:
    rng = np.random.default_rng(42)
    vol = (rng.standard_normal((spatial, spatial, spatial, n_grad)) * 100 + 500).astype(np.float32)

    t0 = time.perf_counter()
    cpu_result = gibbs_cpu(vol)
    cpu_ms = (time.perf_counter() - t0) * 1000

    gpu.fit(vol)
    t0 = time.perf_counter()
    gpu_result = gpu.fit(vol)
    gpu_ms = (time.perf_counter() - t0) * 1000

    speedup = cpu_ms / gpu_ms
    max_diff = float(np.max(np.abs(cpu_result - gpu_result)))
    correct = "PASS" if max_diff < 1.0 else "FAIL"
    print(f"{label:<12} {cpu_ms:>10.1f} {gpu_ms:>10.1f} {speedup:>9.1f}x {max_diff:>12.4f} {correct:>10}")
