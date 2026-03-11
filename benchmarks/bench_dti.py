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


_cpu = _load_module("cpu_dti", "cpu_dti.py")
_gpu = _load_module("gpu_dti", "gpu_dti.py")

cpu_dti_ols = _cpu.cpu_dti_ols
make_w_inv  = _cpu.make_w_inv
GpuDTI      = _gpu.GpuDTI
NUM_DIRS    = _cpu.NUM_DIRS


class TimeDTI:
    """
    Full round-trip benchmark: signal upload → GPU compute → readback.

    Simulates a naïve per-call usage pattern (same as our earlier matmul tests).
    Sizes are realistic DTI brain volumes (100k–1M voxels).
    """

    params = [100_000, 500_000, 1_000_000]
    param_names = ["num_voxels"]
    timeout = 300

    def setup(self, num_voxels):
        rng = np.random.default_rng(0)
        self.signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
        self.w_inv  = make_w_inv()
        self.gpu    = GpuDTI()

    def time_cpu(self, num_voxels):
        cpu_dti_ols(self.w_inv, self.signal)

    def time_gpu(self, num_voxels):
        self.gpu.fit(self.w_inv, self.signal)


class TimeDTICompute:
    """
    Pre-loaded benchmark: signal already resident on GPU, only compute + readback timed.

    This mirrors real-world usage in neuroimaging pipelines where the entire
    brain volume is uploaded once and processed in a single GPU pass —
    exactly the scenario where GPU crushes CPU.
    """

    params = [100_000, 500_000, 1_000_000]
    param_names = ["num_voxels"]
    timeout = 300

    def setup(self, num_voxels):
        rng = np.random.default_rng(0)
        signal      = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
        self.signal = signal
        self.w_inv  = make_w_inv()
        self.gpu    = GpuDTI()
        # Pre-upload both w_inv and signal to GPU memory once
        self.buf_w_inv, self.buf_signal, self.V, self.D, self.C = self.gpu.preload(self.w_inv, signal)

    def time_cpu(self, num_voxels):
        cpu_dti_ols(self.w_inv, self.signal)

    def time_gpu(self, num_voxels):
        self.gpu.dispatch(self.V, self.D, self.C, self.buf_w_inv, self.buf_signal)
