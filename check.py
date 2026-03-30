import numpy as np
import sys
sys.path.insert(0, 'benchmarks')
from nlmeans_wgpu import nlmeans_patch_weights_cpu, nlmeans_patch_weights_gpu

rng = np.random.default_rng(42)

for size in [16, 32]:
    data = (rng.random((size, size, size)) * 100).astype('float32')
    cpu = nlmeans_patch_weights_cpu(data, 1, 3, 50.0)
    gpu = nlmeans_patch_weights_gpu(data, 1, 3, 50.0)

    match = np.allclose(cpu, gpu, atol=1e-3)
    max_diff = float(np.max(np.abs(cpu - gpu)))
    nonzero_gpu = int(np.count_nonzero(gpu))
    total = gpu.size

    print(f"{size}^3: match={match}, max_diff={max_diff:.4f}, gpu_nonzero={nonzero_gpu}/{total}")