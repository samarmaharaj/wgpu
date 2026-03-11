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


cpu_matmul = _load_module("cpu_matmul", "cpu_matmul.py").cpu_matmul
GpuMatMul = _load_module("gpu_matmul", "gpu_matmul.py").GpuMatMul


class TimeMatMul:
    """Full round-trip: upload + compute + readback timed together."""

    params = [256, 512, 1024, 2048, 4096]
    param_names = ["n"]
    timeout = 300

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.a = rng.random((n, n), dtype=np.float32)
        self.b = rng.random((n, n), dtype=np.float32)
        self.gpu = GpuMatMul()

    def time_cpu(self, n):
        cpu_matmul(self.a, self.b)

    def time_gpu(self, n):
        self.gpu.multiply(self.a, self.b)


class TimeMatMulCompute:
    """Pre-loaded: A and B already on GPU. Only compute + readback is timed."""

    params = [256, 512, 1024, 2048, 4096]
    param_names = ["n"]
    timeout = 300

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.a = rng.random((n, n), dtype=np.float32)
        self.b = rng.random((n, n), dtype=np.float32)
        self.gpu = GpuMatMul()
        self.buf_a, self.buf_b, self.M, self.K, self.N = self.gpu.preload(self.a, self.b)

    def time_cpu(self, n):
        cpu_matmul(self.a, self.b)

    def time_gpu(self, n):
        self.gpu.dispatch(self.M, self.K, self.N, self.buf_a, self.buf_b)
