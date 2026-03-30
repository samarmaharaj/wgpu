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


_cpu = _load_module("cpu_dti_wls", "cpu_dti_wls.py")
_gpu = _load_module("gpu_dti_wls", "gpu_dti_wls.py")

cpu_dti_wls = _cpu.cpu_dti_wls
make_dti_design = _cpu.make_dti_design
NUM_DIRS = _cpu.NUM_DIRS
GpuDTIWLS = _gpu.GpuDTIWLS
HAS_WGPU = _gpu.HAS_WGPU


class TimeDTIWLS:
    params = [10_000, 50_000, 100_000]
    param_names = ["num_voxels"]
    timeout = 600

    def setup(self, num_voxels):
        rng = np.random.default_rng(0)
        self.signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
        self.design = make_dti_design()
        self.gpu = GpuDTIWLS() if HAS_WGPU else None

    def time_cpu(self, num_voxels):
        cpu_dti_wls(self.design, self.signal)

    def time_gpu(self, num_voxels):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")
        self.gpu.fit(self.design, self.signal)


class TimeDTIWLSCompute:
    params = [10_000, 50_000, 100_000]
    param_names = ["num_voxels"]
    timeout = 600

    def setup(self, num_voxels):
        if not HAS_WGPU:
            raise NotImplementedError("wgpu not installed")

        rng = np.random.default_rng(0)
        self.signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
        self.design = make_dti_design()
        self.gpu = GpuDTIWLS()
        self.buf_design, self.buf_signal, self.V, self.D, self.C = self.gpu.preload(self.design, self.signal)

    def time_cpu(self, num_voxels):
        cpu_dti_wls(self.design, self.signal)

    def time_gpu(self, num_voxels):
        self.gpu.dispatch(self.V, self.D, self.C, self.buf_design, self.buf_signal)
