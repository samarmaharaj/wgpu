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


cpu_vec_val_vect = _load_module("cpu_vec_val_vect", "cpu_vec_val_vect.py").cpu_vec_val_vect
GpuVecValVect = _load_module("gpu_vec_val_vect", "gpu_vec_val_vect.py").GpuVecValVect


class TimeVecValVect:
    """Full round-trip: upload + compute + readback timed together."""

    params = [100_000, 500_000, 1_000_000]
    param_names = ["n_tensors"]
    timeout = 300

    def setup(self, n_tensors):
        rng = np.random.default_rng(0)
        self.evecs = rng.standard_normal((n_tensors, 3, 3), dtype=np.float32)
        self.evals = rng.uniform(0.1, 3.0, (n_tensors, 3)).astype(np.float32)
        self.gpu = GpuVecValVect()

    def time_cpu(self, n_tensors):
        cpu_vec_val_vect(self.evecs, self.evals)

    def time_gpu(self, n_tensors):
        self.gpu.fit(self.evecs, self.evals)


class TimeVecValVectCompute:
    """Pre-loaded: evecs/evals already on GPU. Only compute + readback is timed."""

    params = [100_000, 500_000, 1_000_000]
    param_names = ["n_tensors"]
    timeout = 300

    def setup(self, n_tensors):
        rng = np.random.default_rng(0)
        self.evecs = rng.standard_normal((n_tensors, 3, 3), dtype=np.float32)
        self.evals = rng.uniform(0.1, 3.0, (n_tensors, 3)).astype(np.float32)
        self.gpu = GpuVecValVect()
        self.buf_evecs, self.buf_evals, self.n = self.gpu.preload(self.evecs, self.evals)

    def time_cpu(self, n_tensors):
        cpu_vec_val_vect(self.evecs, self.evals)

    def time_gpu(self, n_tensors):
        self.gpu.dispatch(self.buf_evecs, self.buf_evals, self.n)
