import argparse
from statistics import mean
from time import perf_counter

import numpy as np

from cpu_vector_add import cpu_vector_add
from gpu_vector_add import GpuVectorAdder


def benchmark(n=1_000_000, repeats=5):
	rng = np.random.default_rng(42)
	a = rng.random(n, dtype=np.float32)
	b = rng.random(n, dtype=np.float32)
	gpu_adder = GpuVectorAdder()

	cpu_result = cpu_vector_add(a, b)
	gpu_result = gpu_adder.add(a, b)

	if not np.allclose(cpu_result, gpu_result, rtol=1e-6, atol=1e-6):
		raise RuntimeError("CPU and GPU results do not match")

	cpu_times = []
	gpu_times = []

	for _ in range(repeats):
		start = perf_counter()
		cpu_vector_add(a, b)
		cpu_times.append(perf_counter() - start)

		start = perf_counter()
		gpu_adder.add(a, b)
		gpu_times.append(perf_counter() - start)

	cpu_mean = mean(cpu_times)
	gpu_mean = mean(gpu_times)

	return {
		"size": n,
		"repeats": repeats,
		"cpu_mean": cpu_mean,
		"gpu_mean": gpu_mean,
		"speedup": cpu_mean / gpu_mean if gpu_mean else float("inf"),
	}


def main():
	parser = argparse.ArgumentParser(description="Compare NumPy CPU vector add with a wgpu compute shader")
	parser.add_argument("--size", type=int, default=1_000_000, help="Number of float32 elements per vector")
	parser.add_argument("--repeats", type=int, default=5, help="How many times to time each implementation")
	args = parser.parse_args()

	results = benchmark(n=args.size, repeats=args.repeats)
	cpu_ms = results["cpu_mean"] * 1000
	gpu_ms = results["gpu_mean"] * 1000

	print(f"Vector size: {results['size']:,}")
	print(f"Repeats: {results['repeats']}")
	print(f"CPU mean: {cpu_ms:.3f} ms")
	print(f"GPU mean: {gpu_ms:.3f} ms")

	if results["speedup"] >= 1:
		print(f"GPU is {results['speedup']:.2f}x faster")
	else:
		print(f"CPU is {1 / results['speedup']:.2f}x faster")


if __name__ == "__main__":
	main()
