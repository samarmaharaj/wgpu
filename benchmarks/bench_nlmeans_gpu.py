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


cpu_nlmeans_patch_weights = _load_module("cpu_nlmeans", "cpu_nlmeans.py").nlmeans_patch_weights_cpu
_gpu_mod = _load_module("gpu_nlmeans", "gpu_nlmeans.py")
GpuNLMeans = _gpu_mod.GpuNLMeans
HAS_WGPU = _gpu_mod.HAS_WGPU


def _make_volume(size):
    rng = np.random.default_rng(42)
    base = np.ones((size, size, size), dtype=np.float32) * 500.0
    noise = rng.standard_normal((size, size, size)).astype(np.float32) * 50.0
    return (base + noise).astype(np.float32)


class TimeNLMeans:
    """Full round-trip: upload + compute + readback timed together."""

    params = [16, 24, 32]
    param_names = ["vol_size"]
    timeout = 300

    PATCH_RADIUS = 1
    BLOCK_RADIUS = 3
    SIGMA = 50.0

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")
        self.data = _make_volume(vol_size)
        self.gpu = GpuNLMeans()

    def time_cpu(self, vol_size):
        cpu_nlmeans_patch_weights(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            block_radius=self.BLOCK_RADIUS,
            sigma=self.SIGMA,
        )

    def time_gpu(self, vol_size):
        self.gpu.fit(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            block_radius=self.BLOCK_RADIUS,
            sigma=self.SIGMA,
        )


class TimeNLMeansCompute:
    """Pre-loaded: padded volume already on GPU. Only compute + readback is timed."""

    params = [16, 24, 32]
    param_names = ["vol_size"]
    timeout = 300

    PATCH_RADIUS = 1
    BLOCK_RADIUS = 3
    SIGMA = 50.0

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuNLMeans()
        self.buf_vol, self.X, self.Y, self.Z = self.gpu.preload(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            block_radius=self.BLOCK_RADIUS,
        )

    def time_cpu(self, vol_size):
        cpu_nlmeans_patch_weights(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            block_radius=self.BLOCK_RADIUS,
            sigma=self.SIGMA,
        )

    def time_gpu(self, vol_size):
        self.gpu.dispatch(
            self.buf_vol,
            self.X,
            self.Y,
            self.Z,
            patch_radius=self.PATCH_RADIUS,
            block_radius=self.BLOCK_RADIUS,
            sigma=self.SIGMA,
        )
