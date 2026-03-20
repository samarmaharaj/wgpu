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


cpu_mod = _load_module("cpu_mppca_full", "cpu_mppca_full.py")
gpu_mod = _load_module("gpu_mppca_full", "gpu_mppca_full.py")

mppca_cpu = cpu_mod.mppca_cpu
GpuMPPCAFull = gpu_mod.GpuMPPCAFull
HAS_WGPU = gpu_mod.HAS_WGPU


def _make_volume(size, channels):
    rng = np.random.default_rng(42)
    return (rng.standard_normal((size, size, size, channels)) * 50.0 + 300.0).astype(np.float32)


class TimeMPPCAFullGPU:
    params = [
        (8, 8),
        (12, 16),
        (16, 16),
    ]
    param_names = ["shape"]
    timeout = 600

    patch_radius = 1

    def setup(self, shape):
        spatial, channels = shape
        self.data = _make_volume(spatial, channels)
        if HAS_WGPU:
            self.gpu = GpuMPPCAFull()

    def time_mppca_cpu(self, shape):
        mppca_cpu(self.data, patch_radius=self.patch_radius)

    def time_mppca_gpu(self, shape):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not available")
        self.gpu.fit(self.data, patch_radius=self.patch_radius)
