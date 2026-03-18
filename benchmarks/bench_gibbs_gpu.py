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


gibbs_suppress_cpu = _load_module("cpu_gibbs", "cpu_gibbs.py").gibbs_suppress_cpu
_gpu_mod = _load_module("gpu_gibbs", "gpu_gibbs.py")
GpuGibbsSuppress = _gpu_mod.GpuGibbsSuppress
HAS_WGPU = _gpu_mod.HAS_WGPU


def _make_volume(size):
    rng = np.random.default_rng(123)
    base = np.full((size, size, size), 1000.0, dtype=np.float32)
    noise = rng.standard_normal((size, size, size)).astype(np.float32) * 80.0
    return base + noise


class TimeGibbs:
    """Full round-trip: upload + compute + readback timed together."""

    params = [48, 64, 80]
    param_names = ["vol_size"]
    timeout = 300

    ALPHA = 0.8

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuGibbsSuppress()

    def time_cpu(self, vol_size):
        gibbs_suppress_cpu(self.data, alpha=self.ALPHA)

    def time_gpu(self, vol_size):
        self.gpu.fit(self.data, alpha=self.ALPHA)


class TimeGibbsCompute:
    """Pre-loaded: volume already on GPU. Only compute + readback is timed."""

    params = [48, 64, 80]
    param_names = ["vol_size"]
    timeout = 300

    ALPHA = 0.8

    def setup(self, vol_size):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        self.data = _make_volume(vol_size)
        self.gpu = GpuGibbsSuppress()
        self.buf_vol, self.x, self.y, self.z = self.gpu.preload(self.data)

    def time_cpu(self, vol_size):
        gibbs_suppress_cpu(self.data, alpha=self.ALPHA)

    def time_gpu(self, vol_size):
        self.gpu.dispatch(self.buf_vol, self.x, self.y, self.z, alpha=self.ALPHA)