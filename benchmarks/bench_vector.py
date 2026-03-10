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


cpu_vector_add = _load_module("cpu_vector_add", "cpu_vector_add.py").cpu_vector_add
GpuVectorAdder = _load_module("gpu_vector_add", "gpu_vector_add.py").GpuVectorAdder


class TimeVectorAdd:

    params = [100_000, 1_000_000, 5_000_000]
    param_names = ["n"]
    timeout = 300

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.a = rng.random(n, dtype=np.float32)
        self.b = rng.random(n, dtype=np.float32)
        self.gpu_adder = GpuVectorAdder()

    def time_cpu(self, n):
        cpu_vector_add(self.a, self.b)

    def time_gpu(self, n):
        self.gpu_adder.add(self.a, self.b)