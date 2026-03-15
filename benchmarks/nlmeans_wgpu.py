from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


CPU_GPU_DIR = Path(__file__).resolve().parent.parent / "cpuGpuTest"


def _load_module(module_name, file_name):
    spec = spec_from_file_location(module_name, CPU_GPU_DIR / file_name)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_cpu = _load_module("cpu_nlmeans", "cpu_nlmeans.py")
_gpu = _load_module("gpu_nlmeans", "gpu_nlmeans.py")

nlmeans_patch_weights_cpu = _cpu.nlmeans_patch_weights_cpu
nlmeans_patch_weights_gpu = _gpu.nlmeans_patch_weights_gpu
GpuNLMeans = _gpu.GpuNLMeans
HAS_WGPU = _gpu.HAS_WGPU
