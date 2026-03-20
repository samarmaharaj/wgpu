"""
Benchmarking and comparison suite for CPU vs GPU MPPCA implementations.

Provides:
- Performance comparison across different volume sizes and parameters
- Correctness validation (CPU vs GPU agreement)
- Memory usage analysis
- Parameter sensitivity analysis
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cpu_mppca_full import mppca_cpu

try:
    from gpu_mppca_full import GpuMPPCAFull, HAS_WGPU
except ImportError:
    HAS_WGPU = False


class MPPCABenchmark:
    """Benchmark runner for MPPCA implementations."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []

    def log(self, msg):
        if self.verbose:
            print(msg)

    def benchmark_cpu(self, data, patch_radius=2, tau_factor=None):
        """Time CPU implementation."""
        t0 = time.perf_counter()
        result = mppca_cpu(data, patch_radius=patch_radius, tau_factor=tau_factor,
                          verbose=False)
        t1 = time.perf_counter()
        return result, (t1 - t0) * 1000  # ms

    def benchmark_gpu(self, data, patch_radius=2, tau_factor=None):
        """Time GPU implementation."""
        if not HAS_WGPU:
            return None, None

        gpu = GpuMPPCAFull()
        t0 = time.perf_counter()
        result = gpu.fit(data, patch_radius=patch_radius, tau_factor=tau_factor)
        t1 = time.perf_counter()
        return result, (t1 - t0) * 1000  # ms

    def compare_implementations(self, data, patch_radius=2, tau_factor=None, atol=1e-1):
        """Compare CPU and GPU results."""
        self.log("Running CPU implementation...")
        result_cpu, time_cpu = self.benchmark_cpu(data, patch_radius, tau_factor)

        if not HAS_WGPU:
            self.log("GPU not available (wgpu not installed)")
            return {
                "cpu_time": time_cpu,
                "gpu_time": None,
                "speedup": None,
                "max_diff": None,
                "allclose": None,
            }

        self.log("Running GPU implementation...")
        result_gpu, time_gpu = self.benchmark_gpu(data, patch_radius, tau_factor)

        # Comparison
        max_diff = np.max(np.abs(result_cpu - result_gpu))
        allclose = np.allclose(result_cpu, result_gpu, atol=atol)
        speedup = time_cpu / time_gpu if time_gpu > 0 else 0

        return {
            "cpu_time": time_cpu,
            "gpu_time": time_gpu,
            "speedup": speedup,
            "max_diff": max_diff,
            "allclose": allclose,
            "result_cpu": result_cpu,
            "result_gpu": result_gpu,
        }

    def print_results(self, name, results):
        """Pretty-print results."""
        print(f"\n{'='*70}")
        print(f"Benchmark: {name}")
        print(f"{'='*70}")
        
        print(f"CPU time:       {results['cpu_time']:>12.2f} ms")
        
        if results['gpu_time'] is not None:
            print(f"GPU time:       {results['gpu_time']:>12.2f} ms")
            print(f"Speedup:        {results['speedup']:>12.2f}×")
            print(f"Max diff:       {results['max_diff']:>12.6e}")
            print(f"Allclose:       {str(results['allclose']):>12}")
        else:
            print("GPU:            Not available")

    def benchmark_scaling(self, base_size=4, max_size=32, n_grad=8):
        """Benchmark scaling with volume size."""
        print("\n" + "="*70)
        print("SCALING ANALYSIS: Volume size vs performance")
        print("="*70)
        print(f"{'Volume':^20} {'CPU (ms)':^15} {'GPU (ms)':^15} {'Speedup':^15}")
        print("-"*70)

        rng = np.random.default_rng(42)

        for size in range(base_size, max_size + 1, 4):
            data = rng.standard_normal((size, size, size, n_grad)).astype(np.float32)
            
            results = self.compare_implementations(data, patch_radius=1)
            
            vol_str = f"{size}³ × {n_grad}"
            cpu_str = f"{results['cpu_time']:.2f}"
            gpu_str = f"{results['gpu_time']:.2f}" if results['gpu_time'] else "N/A"
            speedup_str = f"{results['speedup']:.2f}×" if results['speedup'] else "N/A"
            
            print(f"{vol_str:^20} {cpu_str:>15} {gpu_str:>15} {speedup_str:>15}")

    def benchmark_patch_radius(self, size=16, n_grad=8):
        """Benchmark sensitivity to patch radius."""
        print("\n" + "="*70)
        print("PATCH RADIUS ANALYSIS")
        print("="*70)
        print(f"{'Patch R':^20} {'Patch size':^20} {'CPU (ms)':^15} {'GPU (ms)':^15}")
        print("-"*70)

        rng = np.random.default_rng(42)
        data = rng.standard_normal((size, size, size, n_grad)).astype(np.float32)

        for r in range(1, 5):
            patch_size = (2*r + 1) ** 3
            results = self.compare_implementations(data, patch_radius=r)
            
            cpu_str = f"{results['cpu_time']:.2f}"
            gpu_str = f"{results['gpu_time']:.2f}" if results['gpu_time'] else "N/A"
            
            print(f"{r:^20} {patch_size:^20} {cpu_str:>15} {gpu_str:>15}")

    def benchmark_dimensions(self, size=12):
        """Benchmark sensitivity to number of channels."""
        print("\n" + "="*70)
        print("CHANNEL DIMENSIONALITY ANALYSIS")
        print("="*70)
        print(f"{'Channels':^20} {'Data size (MB)':^20} {'CPU (ms)':^15} {'GPU (ms)':^15}")
        print("-"*70)

        rng = np.random.default_rng(42)

        for n_grad in [4, 8, 16, 32, 48, 64]:
            data = rng.standard_normal((size, size, size, n_grad)).astype(np.float32)
            data_mb = data.nbytes / (1024**2)
            
            results = self.compare_implementations(data, patch_radius=1)
            
            cpu_str = f"{results['cpu_time']:.2f}"
            gpu_str = f"{results['gpu_time']:.2f}" if results['gpu_time'] else "N/A"
            
            print(f"{n_grad:^20} {data_mb:>19.2f} {cpu_str:>15} {gpu_str:>15}")

    def memory_analysis(self, size=16, n_grad=32, patch_radius=2):
        """Analyze memory requirements."""
        print("\n" + "="*70)
        print("MEMORY ANALYSIS")
        print("="*70)

        X, Y, Z, d = size, size, size, n_grad
        v = X * Y * Z
        N = (2 * patch_radius + 1) ** 3

        buffers = {
            "Input volume": v * d * 4,
            "Means": v * d * 4,
            "Covariances": v * d * d * 4,
            "X_centered": v * N * d * 4,
            "Eigenvalues": v * d * 4,
            "Eigenvectors": v * d * d * 4,
            "thetax": v * d * 4,
            "theta": v * 4,
        }

        total = 0
        print(f"{'Buffer':<25} {'Size (MB)':<15} {'Percent':<10}")
        print("-"*50)

        for name, size_bytes in buffers.items():
            size_mb = size_bytes / (1024**2)
            total += size_mb
            percent = 100 * size_mb / total if total > 0 else 0
            print(f"{name:<25} {size_mb:>14.2f} {percent:>9.1f}%")

        print("-"*50)
        print(f"{'TOTAL':<25} {total:>14.2f} MB")

    def correctness_test(self, size=8, n_grad=8):
        """Run correctness tests on small data."""
        print("\n" + "="*70)
        print("CORRECTNESS TEST")
        print("="*70)

        rng = np.random.default_rng(42)
        data = rng.standard_normal((size, size, size, n_grad)).astype(np.float32) * 100 + 500

        results = self.compare_implementations(data, patch_radius=1, atol=1e-1)

        print(f"Volume shape:   {data.shape}")
        print(f"CPU time:       {results['cpu_time']:.2f} ms")
        print(f"GPU time:       {results['gpu_time']:.2f} ms" if results['gpu_time'] else "GPU: N/A")
        print(f"Speedup:        {results['speedup']:.2f}×" if results['speedup'] else "Speedup: N/A")
        print(f"Max difference: {results['max_diff']:.6e}" if results['max_diff'] else "N/A")
        print(f"Allclose:       {results['allclose']}" if results['allclose'] is not None else "N/A")

        if results['allclose'] is False:
            print("\n⚠️  WARNING: CPU and GPU results differ beyond tolerance!")
        elif results['allclose']:
            print("\n✓ CPU and GPU results match!")


def main():
    """Run full benchmark suite."""
    print("\n" + "="*70)
    print("MPPCA CPU vs GPU Benchmark Suite")
    print("="*70)

    bench = MPPCABenchmark(verbose=True)

    # 1. Correctness
    bench.correctness_test(size=8, n_grad=8)

    # 2. Scaling
    bench.benchmark_scaling(base_size=4, max_size=24, n_grad=8)

    # 3. Patch radius sensitivity
    bench.benchmark_patch_radius(size=16, n_grad=8)

    # 4. Dimensionality sensitivity
    bench.benchmark_dimensions(size=12)

    # 5. Memory analysis
    bench.memory_analysis(size=16, n_grad=32, patch_radius=2)

    print("\n" + "="*70)
    print("Benchmark suite completed")
    print("="*70)


if __name__ == "__main__":
    main()
