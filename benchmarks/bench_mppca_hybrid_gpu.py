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


cpu_mppca_hybrid = _load_module("cpu_mppca_hybrid", "cpu_mppca_hybrid.py").mppca_hybrid_cpu
_gpu_mod = _load_module("gpu_mppca_hybrid", "gpu_mppca_hybrid.py")
GpuMPPCAHybrid = _gpu_mod.GpuMPPCAHybrid
HAS_WGPU = _gpu_mod.HAS_WGPU


def _make_volume(size, channels=12):
    rng = np.random.default_rng(7)
    base = np.full((size, size, size, channels), 700.0, dtype=np.float32)
    noise = rng.standard_normal((size, size, size, channels)).astype(np.float32) * 45.0
    return base + noise


class TimeMPPCAHybrid:
    """Hybrid MPPCA: GPU stats + CPU eigh + GPU reconstruction."""

    params = [8, 10, 12]
    param_names = ["vol_size"]
    timeout = 600

    PATCH_RADIUS = 1

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuMPPCAHybrid()

    def time_cpu(self, vol_size):
        cpu_mppca_hybrid(self.data, patch_radius=self.PATCH_RADIUS)

    def time_gpu(self, vol_size):
        self.gpu.fit(self.data, patch_radius=self.PATCH_RADIUS)


class TimeMPPCAHybridCompute:
    """Pre-loaded volume: avoids host->device upload for fair pipeline comparison."""

    params = [8, 10, 12]
    param_names = ["vol_size"]
    timeout = 600

    PATCH_RADIUS = 1

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuMPPCAHybrid()
        self.buf_vol, self.x, self.y, self.z, self.channels = self.gpu.preload(self.data)

    def time_cpu(self, vol_size):
        cpu_mppca_hybrid(self.data, patch_radius=self.PATCH_RADIUS)

    def time_gpu(self, vol_size):
        self.gpu.fit_preloaded(
            self.buf_vol,
            self.x,
            self.y,
            self.z,
            self.channels,
            patch_radius=self.PATCH_RADIUS,
        )