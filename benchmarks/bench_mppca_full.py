"""
ASV Benchmark Suite for MPPCA: CPU vs GPU Full Implementations

Comprehensive benchmarking of:
  - cpu_mppca_full.py (NumPy/SciPy)
  - gpu_mppca_full.py (wgpu + WGSL) - when debugged
  - Scaling analysis across volume sizes and parameters
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np


CPU_GPU_DIR = Path(__file__).resolve().parent.parent / "cpuGpuTest"


def _load_module(module_name, file_name):
    """Dynamically load module from cpuGpuTest directory."""
    spec = spec_from_file_location(module_name, CPU_GPU_DIR / file_name)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Load CPU full implementation
cpu_mod = _load_module("cpu_mppca_full", "cpu_mppca_full.py")
mppca_cpu = cpu_mod.mppca_cpu

# Try loading GPU but gracefully handle if not available
try:
    gpu_mod = _load_module("gpu_mppca_full", "gpu_mppca_full.py")
    GpuMPPCAFull = gpu_mod.GpuMPPCAFull
    HAS_GPU_FULL = gpu_mod.HAS_WGPU
except Exception as e:
    HAS_GPU_FULL = False
    print(f"Note: GPU full implementation not available: {e}")


def _make_volume(size, channels=8):
    """Generate test volume with realistic dMRI-like noise."""
    rng = np.random.default_rng(42)
    base = np.full((size, size, size, channels), 800.0, dtype=np.float32)
    noise = rng.standard_normal((size, size, size, channels)).astype(np.float32) * 60.0
    return base + noise


# ============================================================================
# Benchmark 1: Full Round-Trip (upload + compute + readback)
# ============================================================================

class TimeMPPCAFullRoundTrip:
    """
    Full round-trip CPU vs GPU: upload + compute + readback.
    
    Shows total time including all I/O overhead.
    """

    params = [8, 12, 16, 20]
    param_names = ["vol_size"]
    timeout = 600

    PATCH_RADIUS = 1
    channels = 8

    def setup(self, vol_size):
        self.data = _make_volume(vol_size, self.channels)
        if HAS_GPU_FULL:
            self.gpu = GpuMPPCAFull()

    def time_cpu(self, vol_size):
        """Time CPU full implementation."""
        mppca_cpu(
            self.data,
            patch_radius=self.PATCH_RADIUS,
            verbose=False
        )

    def time_gpu(self, vol_size):
        """Time GPU full implementation (if available)."""
        if not HAS_GPU_FULL:
            raise NotImplementedError("GPU full implementation not available")
        
        self.gpu.fit(
            self.data,
            patch_radius=self.PATCH_RADIUS,
        )


# ============================================================================
# Benchmark 2: CPU-only Scaling Analysis
# ============================================================================

class TimeMPPCACPUScaling:
    """
    CPU performance across different volume sizes.
    
    Helps understand computational complexity scaling.
    """

    params = [8, 12, 16, 20, 24]
    param_names = ["vol_size"]
    timeout = 600

    PATCH_RADIUS = 1
    channels = 8

    def setup(self, vol_size):
        self.data = _make_volume(vol_size, self.channels)

    def time_cpu(self, vol_size):
        """Time CPU implementation."""
        mppca_cpu(self.data, patch_radius=self.PATCH_RADIUS, verbose=False)


# ============================================================================
# Benchmark 3: Parameter Sensitivity - Patch Radius
# ============================================================================

class TimeMPPCAPatchRadius:
    """
    Sensitivity to patch radius on CPU.
    
    Larger patches = more computation but potentially better denoising.
    """

    params = [1, 2, 3]
    param_names = ["patch_radius"]
    timeout = 600

    vol_size = 12
    channels = 8

    def setup(self, patch_radius):
        self.data = _make_volume(self.vol_size, self.channels)

    def time_cpu(self, patch_radius):
        """Time CPU with different patch radii."""
        mppca_cpu(self.data, patch_radius=patch_radius, verbose=False)


# ============================================================================
# Benchmark 4: Parameter Sensitivity - Dimensionality
# ============================================================================

class TimeMPPCADimensionality:
    """
    Sensitivity to number of channels (dimensionality).
    
    Higher dimensions = larger covariance matrices = more computation.
    """

    params = [4, 8, 16, 32, 48]
    param_names = ["channels"]
    timeout = 600

    vol_size = 12
    patch_radius = 1

    def setup(self, channels):
        self.data = _make_volume(self.vol_size, channels)

    def time_cpu(self, channels):
        """Time CPU with different numbers of channels."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)


# ============================================================================
# Benchmark 5: Memory Analysis (CPU)
# ============================================================================

class MemMPPCACPU:
    """
    Memory usage analysis for CPU implementation.
    
    Measured as peak memory during execution.
    """

    params = [8, 12, 16, 20]
    param_names = ["vol_size"]
    timeout = 600

    patch_radius = 1
    channels = 8

    def setup(self, vol_size):
        self.data = _make_volume(vol_size, self.channels)

    def mem_cpu(self, vol_size):
        """Memory used by CPU implementation."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)


# ============================================================================
# Benchmark 6: Speedup Analysis (when GPU available)
# ============================================================================

class TimeMPPCASpeedup:
    """
    Direct speedup measurement: CPU time / GPU time.
    
    Demonstrates GPU acceleration factor.
    """

    params = [12, 16, 20]
    param_names = ["vol_size"]
    timeout = 600

    patch_radius = 1
    channels = 8

    def setup(self, vol_size):
        self.data = _make_volume(vol_size, self.channels)
        if HAS_GPU_FULL:
            self.gpu = GpuMPPCAFull()

    def track_speedup(self, vol_size):
        """Track CPU/GPU speedup ratio."""
        if not HAS_GPU_FULL:
            return np.nan
        
        import time
        
        # Time CPU
        t0 = time.perf_counter()
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)
        t_cpu = time.perf_counter() - t0
        
        # Time GPU
        t0 = time.perf_counter()
        self.gpu.fit(self.data, patch_radius=self.patch_radius)
        t_gpu = time.perf_counter() - t0
        
        return t_cpu / t_gpu if t_gpu > 0 else np.nan


# ============================================================================
# Benchmark 7: Complex Scenario - Realistic dMRI Parameters
# ============================================================================

class TimeMPPCARealistic:
    """
    Benchmark with realistic dMRI parameters.
    
    Typical dMRI data: large volumes, high-dimensional (many b-values/directions)
    """

    params = [
        ("64x64x64_32grad", 64, 32),
        ("64x64x64_64grad", 64, 64),
        ("128x128x128_32grad", 128, 32),
    ]
    param_names = ["scenario"]
    timeout = 1200

    patch_radius = 2

    def setup(self, scenario):
        vol_size, channels = scenario[1:] if len(scenario) > 1 else (64, 32)
        self.data = _make_volume(vol_size, channels)

    def time_cpu(self, scenario):
        """Time CPU on realistic data."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)

    def time_gpu(self, scenario):
        """Time GPU on realistic data (if available)."""
        if not HAS_GPU_FULL:
            raise NotImplementedError("GPU full implementation not available")
        
        gpu = GpuMPPCAFull()
        gpu.fit(self.data, patch_radius=self.patch_radius)


# ============================================================================
# Benchmark 8: Stage-wise Analysis (CPU)
# ============================================================================

class TimeMPPCAStages:
    """
    Detailed timing for each MPPCA stage.
    
    Identifies computational bottlenecks.
    """

    params = [12, 16, 20]
    param_names = ["vol_size"]
    timeout = 600

    patch_radius = 1
    channels = 8

    def setup(self, vol_size):
        self.data = _make_volume(vol_size, self.channels)

    def track_stage_a_time(self, vol_size):
        """Track Stage A (mean + covariance) time."""
        # This would require instrumenting the CPU code
        # For now, we run the full implementation
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)
        return 0.0  # Placeholder

    def track_stage_b_time(self, vol_size):
        """Track Stage B (eigendecomposition) time."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)
        return 0.0  # Placeholder

    def track_stage_c_time(self, vol_size):
        """Track Stage C (reconstruction) time."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)
        return 0.0  # Placeholder

    def track_stage_d_time(self, vol_size):
        """Track Stage D (normalization) time."""
        mppca_cpu(self.data, patch_radius=self.patch_radius, verbose=False)
        return 0.0  # Placeholder
