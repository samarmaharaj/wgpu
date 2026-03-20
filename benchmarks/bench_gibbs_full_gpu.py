from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


CPU_GPU_DIR = Path(__file__).resolve().parent.parent / "cpuGpuTest"


def _load_module(module_name, file_name):
    spec = spec_from_file_location(module_name, CPU_GPU_DIR / file_name)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gibbs_cpu = _load_module("cpu_gibbs_full", "cpu_gibbs_full.py").gibbs_cpu
_gpu_mod = _load_module("gpu_gibbs_full", "gpu_gibbs_full.py")
GpuGibbsFull = _gpu_mod.GpuGibbsFull
HAS_WGPU = _gpu_mod.HAS_WGPU


class BenchGibbsCPU:
    params = [(32, 16), (64, 32), (80, 1)]
    param_names = ["vol_size", "n_gradients"]
    timeout = 600

    def setup(self, params):
        spatial, n_gradients = params
        rng = np.random.default_rng(42)
        self.vol = (rng.standard_normal((spatial, spatial, spatial, n_gradients)) * 100 + 500).astype(np.float32)

    def time_gibbs_cpu(self, params):
        gibbs_cpu(self.vol)


class BenchGibbsGPU:
    params = [(32, 16), (64, 32), (80, 1)]
    param_names = ["vol_size", "n_gradients"]
    timeout = 600

    def setup(self, params):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")
        spatial, n_gradients = params
        rng = np.random.default_rng(42)
        self.vol = (rng.standard_normal((spatial, spatial, spatial, n_gradients)) * 100 + 500).astype(np.float32)
        self.gpu = GpuGibbsFull()
        self.gpu.fit(self.vol)

    def time_gibbs_gpu(self, params):
        self.gpu.fit(self.vol)
