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


cpu_mppca_proxy = _load_module("cpu_mppca", "cpu_mppca.py").mppca_proxy_cpu
_gpu_mod = _load_module("gpu_mppca", "gpu_mppca.py")
GpuMPPCAProxy = _gpu_mod.GpuMPPCAProxy
HAS_WGPU = _gpu_mod.HAS_WGPU


def _make_volume(size, channels=16):
    rng = np.random.default_rng(42)
    base = np.full((size, size, size, channels), 800.0, dtype=np.float32)
    noise = rng.standard_normal((size, size, size, channels)).astype(np.float32) * 60.0
    return base + noise


class TimeMPPCA:
    """Full round-trip: upload + compute + readback timed together."""

    params = [12, 16, 20]
    param_names = ["vol_size"]
    timeout = 300

    PATCH_RADIUS = 1
    TAU = 1.2

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuMPPCAProxy()

    def time_cpu(self, vol_size):
        cpu_mppca_proxy(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            tau=self.TAU,
        )

    def time_gpu(self, vol_size):
        self.gpu.fit(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            tau=self.TAU,
        )


class TimeMPPCACompute:
    """Pre-loaded: volume already on GPU. Only compute + readback is timed."""

    params = [12, 16, 20]
    param_names = ["vol_size"]
    timeout = 300

    PATCH_RADIUS = 1
    TAU = 1.2

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuMPPCAProxy()
        self.buf_vol, self.x, self.y, self.z, self.channels = self.gpu.preload(self.data)

    def time_cpu(self, vol_size):
        cpu_mppca_proxy(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            tau=self.TAU,
        )

    def time_gpu(self, vol_size):
        self.gpu.dispatch(
            self.buf_vol,
            self.x,
            self.y,
            self.z,
            self.channels,
            patch_radius=self.PATCH_RADIUS,
            tau=self.TAU,
        )