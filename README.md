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

---

## March 16, 2026 — 4-candidate ASV run (same env, fixed seeds)

## Experiment 4 — NLM Denoising (`bench_nlmeans_gpu.py`)

Benchmarked using ASV (same tool used by DIPY officially).

```bash
cd C:\dev\GSOC2026\WGPU\wgpu
asv run --dry-run --show-stderr --python=same --quick -b TimeNLMeans
```
Benchmarks run with:

```bash
asv run --dry-run --show-stderr --python=same --quick -b TimeNLMeansCompute
```

### Full round-trip (upload + compute + readback)
| Volume | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| 16³ | 9.26 s | 552 ms | 16.8× |
| 24³ | 30.4 s | 547 ms | 55.6× |
| 32³ | 76.0 s | 532 ms | **142×** |

### Pre-loaded (compute + readback only)
| Volume | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| 16³ | 9.47 s | 582 ms | 16.3× |
| 24³ | 30.9 s | 496 ms | 62.3× |
| 32³ | 76.2 s | 496 ms | **154×** |

**Key insight:** GPU time stays flat at ~500ms regardless of volume 
size — all voxels compute simultaneously. CPU scales as O(N³).
Correctness verified: np.allclose(atol=1e-3) passes for all sizes.

---

Benchmarks run with:

```bash
conda run -n wgpu asv run --dry-run --show-stderr --python=same --quick \
	-b "TimeDTI|TimeDTICompute|TimeNLMeans|TimeNLMeansCompute|TimeVecValVect|TimeVecValVectCompute|TimeSetNumberOfPoints|TimeSetNumberOfPointsCompute"
```

### Candidate summary (largest successful size)

| Candidate | Full round-trip (CPU vs GPU) | Pre-loaded (CPU vs GPU) | Approx speedup (GPU/CPU) |
|---|---|---|---|
| DTI OLS (`500,000` voxels) | `109 ms` vs `331 ms` | `119 ms` vs `245 ms` | `0.33×` full, `0.49×` preloaded |
| NLM (`32^3`) | `1.27 min` vs `547 ms` | `1.27 min` vs `523 ms` | `~139×` full, `~145×` preloaded |
| vec_val_vect (`1,000,000` tensors) | `174 ms` vs `291 ms` | `173 ms` vs `235 ms` | `0.60×` full, `0.74×` preloaded |
| set_number_of_points (`100,000` streamlines) | `31.1 s` vs `375 ms` | `28.5 s` vs `239 ms` | `~83×` full, `~119×` preloaded |

### Correctness checks (fixed RNG seed)

- DTI: `allclose=True`, max abs diff `2.09e-07` (`atol=2e-4`)
- NLM: `allclose=True`, max abs diff `8.39e-05` (`atol=1e-3`)
- vec_val_vect: `allclose=True`, max abs diff `3.81e-06` (`atol=1e-4`)
- set_number_of_points: `allclose=True`, max abs diff `3.81e-06` (`atol=1e-3`)

### Ranking for this hardware/config

1. **NLM** — strongest speedup and stable correctness.
2. **set_number_of_points** — very large speedup, low numerical error.
3. **vec_val_vect** — GPU slower than CPU; low implementation risk but weak benefit.
4. **DTI OLS (current prototype)** — GPU slower and fails at `1,000,000` due per-buffer size limit (`268,435,456` bytes on this backend).

---

## March 19, 2026 — NLM vs MPPCA vs GIBBS (test-data comparison)

This section records the newest findings from the dedicated comparison runner:

```bash
cd C:\dev\GSOC2026\WGPU\wgpu\cpuGpuTest
python compare_mppca_gibbs_nlm.py --repeats 3
```

### Runtime (mean over repeats, fixed synthetic test data)

| Algorithm | CPU | GPU | CPU/GPU speedup |
|---|---:|---:|---:|
| NLM | 19,519.273 ms | 23.352 ms | **835.86×** |
| MPPCA (proxy) | 1,985.398 ms | 5.737 ms | **346.09×** |
| GIBBS (proxy) | 555.297 ms | 4.399 ms | **126.23×** |

### Fastest by absolute GPU runtime

1. **GIBBS** — `4.399 ms`
2. **MPPCA** — `5.737 ms`
3. **NLM** — `23.352 ms`

### Float32 error count vs float64 reference

| Algorithm | Tolerance (`atol`) | CPU32 allclose | CPU32 exceed count | GPU32 allclose | GPU32 exceed count |
|---|---:|---|---:|---|---:|
| NLM | `1e-3` | True | 0 | True | 0 |
| MPPCA (proxy) | `2e-3` | True | 0 | True | 0 |
| GIBBS (proxy) | `1e-4` | False | 9,382 | False | 7,664 |

Observed max absolute differences:
- NLM GPU32 vs float64: `8.575e-04`
- MPPCA GPU32 vs float64: `3.210e-04`
- GIBBS GPU32 vs float64: `3.718e-04`

**Interpretation:** At a strict `1e-4` threshold, GIBBS fails allclose for both CPU32 and GPU32 against float64 reference. This is a tolerance-setting issue, not a GPU-only mismatch.

---

## March 19, 2026 — ASV quick check for `nlmeans|mppca|gibbs`

Executed from repo root with existing environment:

```bash
cd C:\dev\GSOC2026\WGPU\wgpu
python -m asv run --config asv.conf.json \
	-E existing:<python_path> --dry-run --quick -b "nlmeans|mppca|gibbs"
```

ASV discovered and ran all targeted classes successfully:
- `bench_nlmeans_gpu.TimeNLMeans` and `TimeNLMeansCompute`
- `bench_mppca_gpu.TimeMPPCA` and `TimeMPPCACompute`
- `bench_gibbs_gpu.TimeGibbs` and `TimeGibbsCompute`

Representative trend from ASV quick output:
- **NLM:** largest speedup at larger volume sizes.
- **MPPCA (proxy):** strong GPU advantage, with stable sub-second GPU timings in this configuration.
- **GIBBS (proxy):** fastest absolute GPU kernel time among the three, while still showing large CPU/GPU speedup.

> Note: current `MPPCA` and `GIBBS` implementations in `cpuGpuTest` are **GPU-oriented proxies for benchmarking workflow validation**, not full one-to-one reproductions of the exact DIPY production algorithm internals.

---

## March 19, 2026 — Full ASV rerun (all CPU-vs-GPU benchmarks)

To rerun **all** CPU-vs-GPU benchmarks currently in this repo from one command (stable mode, no `--quick`):

```bash
cd C:\dev\GSOC2026\WGPU\wgpu
python -m asv run --config asv.conf.json \
	-E existing:<python_path> --dry-run \
	-b "bench_(vector|matmul|dti|nlmeans_gpu|vec_val_vect|set_number_of_points|mppca_gpu|gibbs_gpu).*time_(cpu|gpu)"
```

Fast smoke-test variant (less stable, but quicker):

```bash
python -m asv run --config asv.conf.json \
	-E existing:<python_path> --dry-run --quick \
	-b "bench_(vector|matmul|dti|nlmeans_gpu|vec_val_vect|set_number_of_points|mppca_gpu|gibbs_gpu).*time_(cpu|gpu)"
```

Example `<python_path>` used in this run:

```bash
c:/Users/samar kumar/.conda/envs/hh-dev/python.exe
```

### Consolidated summary (largest tested size per benchmark)

| Benchmark | Full round-trip (CPU vs GPU) | Pre-loaded (CPU vs GPU) | CPU/GPU speedup (full, preloaded) |
|---|---|---|---|
| Vector add (`n=5,000,000`) | `5.54 ms` vs `305 ms` | N/A | `0.02×`, N/A |
| Matmul tiled (`n=4096`) | `224 ms` vs `1.30 s` | `439 ms` vs `1.14 s` | `0.17×`, `0.39×` |
| DTI OLS (`1,000,000` voxels) | `423 ms` vs `399 ms` | `431 ms` vs `283 ms` | `1.06×`, `1.52×` |
| NLM (`32^3`) | `1.68 min` vs `442 ms` | `1.70 min` vs `401 ms` | `~228×`, `~254×` |
| vec_val_vect (`1,000,000` tensors) | `244 ms` vs `257 ms` | `245 ms` vs `195 ms` | `0.95×`, `1.26×` |
| set_number_of_points (`100,000` streamlines) | `35.0 s` vs `344 ms` | `35.1 s` vs `216 ms` | `~102×`, `~162×` |
| MPPCA proxy (`20^3x16`) | `4.17 s` vs `9.07 ms` | `4.15 s` vs `7.77 ms` | `~460×`, `~534×` |
| GIBBS proxy (`80^3`) | `1.70 s` vs `8.93 ms` | `1.69 s` vs `7.20 ms` | `~190×`, `~235×` |

### Rerun ranking by speedup on this hardware

1. **MPPCA proxy** (after pipeline caching)
2. **NLM**
3. **GIBBS proxy** (after pipeline caching)
4. **set_number_of_points**
5. **DTI OLS** (moderate gain)
6. **vec_val_vect** (mixed; wins only in preloaded mode)
7. **matmul tiled** (GPU slower)
8. **vector add** (GPU much slower)

> Note: after GPU pipeline/shader caching was added for `MPPCA` and `GIBBS`, non-quick ASV reflects their expected high speedups again. If you see unexpectedly low speedups in quick mode, verify with the stable non-quick command above.

---

## March 19, 2026 — Post-fix validation for `MPPCA` and `GIBBS`

After caching shader/pipeline setup in GPU dispatch paths (`gpu_mppca.py`, `gpu_gibbs.py`), we re-ran ASV in non-quick mode:

```bash
cd C:\dev\GSOC2026\WGPU\wgpu
python -m asv run --config asv.conf.json \
	-E existing:<python_path> --dry-run \
	-b "bench_(mppca_gpu|gibbs_gpu).*time_(cpu|gpu)"
```

Largest-size results from this run:

| Benchmark | Full round-trip (CPU vs GPU) | Pre-loaded (CPU vs GPU) | CPU/GPU speedup (full, preloaded) |
|---|---|---|---|
| MPPCA proxy (`20^3x16`) | `4.17 s` vs `9.07 ms` | `4.15 s` vs `7.77 ms` | `~460×`, `~534×` |
| GIBBS proxy (`80^3`) | `1.70 s` vs `8.93 ms` | `1.69 s` vs `7.20 ms` | `~190×`, `~235×` |

This explains the discrepancy with earlier low (~4–12×) ASV quick numbers: those were dominated by per-call compile/setup overhead before pipeline caching.
