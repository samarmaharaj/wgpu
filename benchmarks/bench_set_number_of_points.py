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


_cpu_mod = _load_module("cpu_set_number_of_points", "cpu_set_number_of_points.py")
cpu_set_number_of_points = _cpu_mod.cpu_set_number_of_points
GpuSetNumberOfPoints = _load_module("gpu_set_number_of_points", "gpu_set_number_of_points.py").GpuSetNumberOfPoints


def _make_streamlines(n_streamlines, n_in_points, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_streamlines, n_in_points, 3), dtype=np.float32)
    streamlines = np.cumsum(steps, axis=1, dtype=np.float32)
    return streamlines


class TimeSetNumberOfPoints:
    """Full round-trip: upload + compute + readback timed together."""

    params = [10_000, 50_000, 100_000]
    param_names = ["n_streamlines"]
    timeout = 300

    N_IN = 32
    N_OUT = 50

    def setup(self, n_streamlines):
        self.streamlines = _make_streamlines(n_streamlines, self.N_IN)
        self.gpu = GpuSetNumberOfPoints()

    def time_cpu(self, n_streamlines):
        cpu_set_number_of_points(self.streamlines, nb_points=self.N_OUT)

    def time_gpu(self, n_streamlines):
        self.gpu.fit(self.streamlines, nb_points=self.N_OUT)


class TimeSetNumberOfPointsCompute:
    """Pre-loaded: points/cumulative-lengths on GPU. Only compute + readback is timed."""

    params = [10_000, 50_000, 100_000]
    param_names = ["n_streamlines"]
    timeout = 300

    N_IN = 32
    N_OUT = 50

    def setup(self, n_streamlines):
        self.streamlines = _make_streamlines(n_streamlines, self.N_IN)
        self.gpu = GpuSetNumberOfPoints()
        self.state = self.gpu.preload(self.streamlines, nb_points=self.N_OUT)

    def time_cpu(self, n_streamlines):
        cpu_set_number_of_points(self.streamlines, nb_points=self.N_OUT)

    def time_gpu(self, n_streamlines):
        self.gpu.dispatch(*self.state)
