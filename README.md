# wgpu — CPU vs GPU Benchmark Study

Exploring GPU acceleration of scientific computing workloads using [wgpu-py](https://github.com/pygfx/wgpu-py) (WebGPU for Python) with benchmarks run via [ASV](https://asv.readthedocs.io/). Motivated by potential GPU acceleration of [DIPY](https://dipy.org/) algorithms.

Benchmarks were run on a single commit using an existing Python environment. Each test covers two scenarios:
- **Full round-trip** — includes data upload to GPU, compute dispatch, and readback to CPU.
- **Pre-loaded (Compute only)** — data is already resident on the GPU; only compute + readback is timed.

---

## Experiment 1 — Vector Addition (`bench_vector.py`)

The simplest possible kernel: `c[i] = a[i] + b[i]` for float32 arrays.

| n | CPU (`numpy.add`) | GPU (WebGPU) | Winner |
|---|---|---|---|
| 100,000 | 13.8 ± 0.2 µs | 8.28 ± 0.2 ms | CPU **~600×** |
| 1,000,000 | 847 ± 40 µs | 12.4 ± 0.2 ms | CPU **~15×** |
| 5,000,000 | 5.07 ± 0.09 ms | 37.2 ± 0.7 ms | CPU **~7×** |

**Verdict: CPU wins decisively.**

The GPU times are dominated entirely by the round-trip overhead (buffer allocation, PCIe upload, dispatch, readback). Vector addition is trivially simple — one addition per element — so there is not enough arithmetic to amortize the ~8 ms GPU overhead. NumPy with SIMD is unbeatable here.

**Key insight:** GPU overhead (~8 ms) is constant regardless of workload size. For GPU to win, the kernel must do enough *work per byte* that this fixed cost becomes negligible.

---

## Experiment 2 — Matrix Multiplication (`bench_matmul.py`)

Square matrix multiply: `C = A @ B` for float32 `n×n` matrices.

### 2a. Naive WGSL shader

| n | CPU (numpy/BLAS) | GPU (naive) | Winner |
|---|---|---|---|
| 256 | 211 ± 5 µs | 13.8 ± 3 ms | CPU **~65×** |
| 512 | 987 ± 100 µs | 16.3 ± 4 ms | CPU **~17×** |
| 1024 | 7.02 ± 0.3 ms | 38.9 ± 4 ms | CPU **~5.5×** |

The gap closes rapidly with size — from 65× at n=256 down to 5.5× at n=1024 — but CPU still leads because:
- NumPy uses BLAS (OpenBLAS/MKL) with SIMD and multi-threading.
- The naive shader makes `O(n³)` global memory accesses with no caching.

### 2b. Tiled shared-memory WGSL shader

The shader was upgraded to load 16×16 tiles of A and B into workgroup shared memory, reducing global memory traffic by ~16×.

#### Full round-trip

| n | CPU | GPU (tiled) | Winner |
|---|---|---|---|
| 256 | 197 µs | 8.21 ms | CPU **42×** |
| 512 | 874 µs | 10.3 ms | CPU **12×** |
| 1024 | 6.50 ms | 25.5 ms | CPU **3.9×** |
| 2048 | 39.9 ms | 102 ms | CPU **2.6×** |
| 4096 | 241 ms | 730 ms | CPU **3.0×** |

#### Pre-loaded (compute + readback only)

| n | CPU | GPU (tiled) | Winner |
|---|---|---|---|
| 256 | 188 µs | 7.83 ms | CPU **42×** |
| 512 | 884 µs | 9.78 ms | CPU **11×** |
| 1024 | 4.86 ms | 18.4 ms | CPU **3.8×** |
| 2048 | 31.5 ms | 90.7 ms | CPU **2.9×** |
| 4096 | 259 ms | 648 ms | CPU **2.5×** |

**Verdict: CPU still wins, but tiling improved GPU by ~1.5× (e.g. 1024: 38.9 ms → 25.8 ms).**

The remaining bottleneck is the readback (PCIe GPU→CPU). At n=4096 we transfer 3 × 64 MB across PCIe, which accounts for most of the 730 ms. Pre-loading the inputs saved only 10–12% because the output readback is unavoidable.

**Key insight:** For GPU to beat NumPy/BLAS at general matmul, results must stay on-device across chained operations (as in PyTorch/ML inference pipelines), eliminating all but the final readback.

---

## Experiment 3 — DTI OLS Fitting (`bench_dti.py`)

Diffusion Tensor Imaging (DTI) Ordinary Least Squares fit — a foundational DIPY algorithm. For each voxel we solve:

```
X = W_inv @ ln(S)
```

where `W_inv` is a pre-computed 7×90 pseudo-inverse design matrix and `S` is the 90-direction diffusion signal for that voxel. This is applied independently to every voxel in a brain volume (up to ~2 million voxels in practice).

**GPU strategy:** 1 thread = 1 voxel. The 7×90 design matrix (~2.5 KB) fits entirely in GPU L1 cache and is shared by all threads. Each thread independently loops over 90 gradient directions — zero inter-thread communication, zero branch divergence, perfectly coalesced memory access.

### Full round-trip (upload + compute + readback)

| Voxels | CPU (numpy) | GPU (WebGPU) | Winner |
|---|---|---|---|
| 100,000 | 26.5 ms | 25.7 ms | **GPU ~1.03×** |
| 500,000 | 131 ms | 112 ms | **GPU ~1.17×** |
| 1,000,000 | 263 ms | 221 ms | **GPU ~1.19×** |

### Pre-loaded (signal already on GPU, compute + readback only)

| Voxels | CPU (numpy) | GPU (WebGPU) | Winner |
|---|---|---|---|
| 100,000 | 25.3 ms | 15.1 ms | **GPU ~1.68×** |
| 500,000 | 129 ms | 69.6 ms | **GPU ~1.85×** |
| 1,000,000 | 298 ms | 158 ms | **GPU ~1.89×** |

**Verdict: GPU wins across every single test.**

Even with full round-trip PCIe overhead, the GPU wins because the per-voxel work (90 `log()` ops + 630 multiply-adds) is compute-intensive enough to amortize the upload cost. In the pre-loaded scenario — which mirrors how a real neuroimaging pipeline would work (upload the entire brain volume once, run the fit, read back only the tensor maps) — the GPU is nearly **2× faster** at 1M voxels and the margin is still growing.

---

## Summary of Findings

| Benchmark | Compute intensity | PCIe dominates? | Winner |
|---|---|---|---|
| Vector addition | Very low (1 op/element) | Yes | CPU (~600×) |
| Matrix multiply (naive) | Medium | Yes | CPU (~5–65×) |
| Matrix multiply (tiled) | Medium-high | Yes | CPU (~3–42×) |
| DTI OLS fit (round-trip) | High (630 MADs + 90 logs/voxel) | No | **GPU (~1.2×)** |
| DTI OLS fit (pre-loaded) | High | No | **GPU (~1.9×)** |

**The GPU wins when:**
1. The kernel is compute-intensive enough per element that the PCIe cost is amortized.
2. Data is pre-resident on the GPU across multiple operations (no repeated upload/readback).
3. The workload maps to massively parallel independent threads (1 thread = 1 voxel).

These results directly motivate GPU-accelerating DIPY algorithms like DTI fitting, CSD, and tractography using `wgpu-py`.
