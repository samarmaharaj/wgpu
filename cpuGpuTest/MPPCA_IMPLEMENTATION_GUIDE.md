# MPPCA (Marchenko-Pastur PCA) Full Implementation Guide

## Overview

This documentation covers the complete implementation of the Marchenko-Pastur Principal Component Analysis (MPPCA) denoising algorithm with both CPU and GPU backends.

**Key Files:**
- [cpu_mppca_full.py](cpu_mppca_full.py) — Pure NumPy/SciPy CPU implementation
- [gpu_mppca_full.py](gpu_mppca_full.py) — wgpu-based GPU implementation with WGSL shaders

## What is MPPCA?

**MPPCA** is a denoising algorithm for multi-channel (diffusion) data that:
1. Extracts patches around each voxel
2. Computes patch covariance matrix (per-voxel statistics)
3. Performs eigendecomposition to identify signal vs. noise subspaces
4. Uses **Marchenko-Pastur distribution** to automatically set the noise threshold
5. Reconstructs denoised patches by projecting onto signal subspace
6. Averages overlapping reconstructions at each voxel

**Applications:**
- dMRI/DTI denoising (medical imaging)
- Multi-channel data preprocessing
- Noise suppression in high-dimensional acquisitions

---

## Algorithm Stages

### Stage A: Mean & Covariance Computation

#### Purpose
Extract mean and compute per-voxel covariance matrices from overlapping patches.

#### CPU Implementation
```python
# For each voxel:
#  1. Extract patch centered at voxel (reflection boundary handling)
#  2. Compute mean: μ = (1/N) Σ x_i
#  3. Center: X_centered = X - μ
#  4. Covariance: C = (1/N) X_centered.T @ X_centered
```

**Key details:**
- Patches have size $(2r+1)^3$ where $r$ is `patch_radius`
- Reflection boundary handling prevents artifacts at edges
- Covariance is symmetric, shape $(d \times d)$ where $d$ = number of channels

#### GPU Implementation (WGSL)
Three separate compute shaders:
1. **stage_a_means** — Compute mean over patch
2. **stage_a_centered** — Extract centered patch data
3. **stage_a_cov** — Compute covariance matrix

Each kernel runs in parallel across all voxels (workgroup_size=256).

**GPU Memory Layout:**
- `means_buf`: size $= |V| \times d \times 4$ bytes
- `covs_buf`: size $= |V| \times d^2 \times 4$ bytes  
- `x_centered_buf`: size $= |V| \times N \times d \times 4$ bytes

where $|V|$ = number of voxels, $N$ = patch samples $(2r+1)^3$

---

### Stage B: Eigendecomposition

#### Purpose
Compute eigenvalues and eigenvectors of each per-voxel covariance matrix.

#### CPU Implementation
```python
# Eigendecomposition: C = W @ diag(λ) @ W.T
# CPU uses: scipy.linalg.eigh (optimized LAPACK)
evals, evecs = eigh(C)
```

**Advantages:**
- Highly optimized (MKL/OpenBLAS backend)
- Automatic sorting in ascending order
- Numerically stable

#### GPU Implementation (WGSL)
**One-sided Jacobi SVD** algorithm:
```
for sweep = 1 to MAX_SWEEPS do
  for all pairs (p, q), p < q do
    Compute rotation θ_{pq} = atan(2C_{pq} / (C_{qq} - C_{pp}))
    Apply rotation: C' = R(θ)^T @ C @ R(θ)
    Accumulate: W = W @ R(θ)
  end for
end for
Extract eigenvalues from diagonal of final C'
```

**Stage B Constants:**
```wgsl
override MAX_SWEEPS : u32;  // Typically 15 sweeps
```

**Sorting:**
After convergence, eigenvalues are sorted in ascending order (insertion sort).

**Why Jacobi?**
- Symmetric matrix eigendecomposition (guaranteed convergence)
- Parallelizable per-pair rotations
- Numerically stable for small matrices

---

### Stage C: Marchenko-Pastur Thresholding & Reconstruction

#### Purpose
Identify the noise dimension using Marchenko-Pastur theory, project onto signal subspace, and reconstruct.

#### Marchenko-Pastur Classifier

The algorithm automatically determines $n_{\text{comp}}$ (number of signal components) by:

```
τ = (4/√Q) × var  where Q = #samples/#channels
Remove eigenvalues λ_i such that λ_i < τ
Keep largest eigenvalues above threshold
```

**CPU Implementation:**
```python
def pca_classifier_ncomps(evals, num_samples, dim):
    c = dim - 1
    var = np.mean(evals)
    tau = 4.0 * np.sqrt((c + 1) / num_samples) * var
    
    while evals[c] < tau and c > 0:
        c -= 1
        var = np.mean(evals[:c])
    
    return c + 1  # Number of signal components
```

#### Reconstruction
For each patch sample:
1. Project onto signal subspace: $\hat{x} = W_{\text{sig}} W_{\text{sig}}^T x$
2. Weight by $\theta = \frac{1}{1 + \text{rank}_{\text{noise}}}$
3. Accumulate in weighted sum

**GPU Implementation (Stage C):**
```wgsl
// For each voxel's patch:
for s in samples:
    // Project: W_proj.T @ x_centered
    proj = W_proj^T @ x_centered[s]
    // Reconstruct: W_proj @ proj + mean
    x_est = W_proj @ proj + μ
    // Accumulate at patch locations
    thetax += x_est * weight
    theta  += weight
end
```

**Why weighted averaging?**
- Overlapping patches at boundaries need consensus
- Weight reflects eigenvalue ratio (signal vs. noise dimensionality)
- Numerically stable normalization

---

### Stage D: Final Normalization

#### Purpose
Normalize accumulated weighted sums to produce final denoised volume.

#### Implementation
```python
# For each voxel:
if theta > 0:
    denoised = thetax / theta
else:
    denoised = original  # Fallback if no patches
```

#### GPU Implementation
```wgsl
@compute @workgroup_size(256)
fn stage_d_normalize(@builtin(global_invocation_id) gid : vec3<u32>) {
    let t = theta[idx];
    if t > 0.0:
        denoised[idx] = thetax[idx] / t;
    else:
        denoised[idx] = original[idx];
}
```

---

## CPU vs GPU Trade-offs

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Speed** | Baseline (1×) | 10-100× faster for large volumes |
| **Memory** | O(d²) per voxel on-fly | All intermediate buffers on device |
| **Precision** | float64 possible | float32 only |
| **Customization** | Easy (NumPy) | Requires WGSL shader edits |
| **Dependencies** | NumPy, SciPy | wgpu, Vulkan/Metal/DX12 runtime |
| **Debugging** | Standard tools | Limited (shader debugging) |

---

## Usage Examples

### CPU Implementation
```python
from cpu_mppca_full import mppca_cpu
import numpy as np

# Load or generate data: shape (X, Y, Z, channels)
data = np.random.randn(64, 64, 64, 32).astype(np.float32)

# Run MPPCA
denoised = mppca_cpu(
    data,
    patch_radius=2,      # Patch size: (2*2+1)^3 = 5^3 = 125 samples
    tau_factor=1.5,      # Marchenko-Pastur threshold multiplier (optional)
    verbose=True
)

print(denoised.shape)  # (64, 64, 64, 32)
```

### GPU Implementation
```python
from gpu_mppca_full import GpuMPPCAFull
import numpy as np

# Allocate GPU class
gpu = GpuMPPCAFull()

# Method 1: One-shot (automatic upload/compute/readback)
denoised = gpu.fit(data, patch_radius=2)

# Method 2: Preload then compute (for repeated operations)
buf_vol, shape = gpu.preload(data, patch_radius=2)
denoised = gpu.fit_preloaded(buf_vol, shape[0], shape[1], shape[2], shape[3])
```

---

## Performance Characteristics

### GPU Memory Requirements
For volume $(X, Y, Z)$ with $d$ channels and patch radius $r$:

| Buffer | Memory |
|--------|--------|
| Input volume | $X \times Y \times Z \times d \times 4$ |
| Means | $X \times Y \times Z \times d \times 4$ |
| Covariances | $X \times Y \times Z \times d^2 \times 4$ |
| X_centered | $X \times Y \times Z \times (2r+1)^3 \times d \times 4$ |
| Eigenvalues | $X \times Y \times Z \times d \times 4$ |
| Eigenvectors | $X \times Y \times Z \times d^2 \times 4$ |
| thetax, theta | $X \times Y \times Z \times (d+1) \times 4$ |
| **Total** | ~$(5d^2 + 2(2r+1)^3 \times d) \times |V| \times 4$ bytes |

### Computational Complexity

| Stage | CPU | GPU |
|-------|-----|-----|
| A (Mean/Cov) | O($(2r+1)^3 \times d^2$) per voxel | $O(1)$ per voxel (parallelized) |
| B (Eigen) | O($d^3 \times \text{sweeps}$) per voxel | O($d^3 \times \text{sweeps}$) per voxel |
| C (Reconstruct) | O($(2r+1)^3 \times d^2$) per voxel | O(1) per voxel (parallelized) |
| D (Normalize) | O($d$) per voxel | O(1) per voxel (parallelized) |

**Key insight:** Stages A, C, D are embarrassingly parallel on GPU (no inter-voxel dependencies). Only Stage B requires computation per voxel but is still 10-100× faster with SIMD/vector operations.

---

## Testing & Validation

### Test Suite
Both implementations include correctness tests:

```bash
# CPU unit tests
python cpu_mppca_full.py

# GPU validation against CPU
python gpu_mppca_full.py
```

**Test Cases:**
- Small volumes (8³, 12³) with 8-16 channels
- Comparison of CPU vs GPU outputs
- Timing measurements
- Tolerance check: `atol=1e-1` (float32 rounding)

### Example Test Output
```
GPU MPPCA tests completed
Test: volume 8^3 with 8 gradients
  CPU time: 245 ms
  GPU time: 18 ms
  Speedup: 13.6×
  Max diff: 0.08
  Allclose (atol=1e-1): True
```

---

## Advanced Configuration

### Custom Tau Factor
The tau factor controls the Marchenko-Pastur threshold:

```python
# Default: tau_factor = 1.0 + √(d/N)
# Lower → more aggressive denoising (risk of signal loss)
# Higher → more conservative (retains more noise)

denoised = mppca_cpu(data, tau_factor=1.2)  # More aggressive
denoised = mppca_cpu(data, tau_factor=2.0)  # More conservative
```

### Advanced GPU Configuration
```python
gpu = GpuMPPCAFull(device=custom_device)  # Custom wgpu device

# Access internal buffers for advanced pipelines
buf_vol, shape = gpu.preload(data)
# ... perform custom operations on GPU ...
```

---

## Known Limitations

### CPU
- Slower for large volumes (X, Y, Z > 50)
- High memory usage if $d$ is large
- Single-threaded (no OpenMP parallelization in current version)

### GPU
- float32 precision only (slight numerical differences vs CPU)
- Requires wgpu-compatible GPU (Vulkan/Metal/DirectX)
- WGSL shader limits on local array sizes (currently ~64 dimensions max)
- No float64 support (data cast to float32)

### Both
- MPPCA assumes noise is Gaussian and isotropic
- Requires sufficient samples: $(2r+1)^3 \gg d$
- May oversmooth if tau_factor is too high

---

## References & Theory

### Marchenko-Pastur Distribution
The Marchenko-Pastur law describes the limiting distribution of eigenvalues of sample covariance matrices from random matrices:

$$\lambda_{\text{MP}} \in [\lambda_-, \lambda_+] \text{ where } \lambda_{\pm} = \sigma^2(1 \pm \sqrt{Q})^2$$

with $Q = \frac{\text{#channels}}{\text{#samples}}$ and $\sigma^2$ = noise variance.

### PCA Denoising
The algorithm projects pure noise into lower-dimensional signal subspace, exploiting the spectral gap between signal and noise eigenvalues.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'wgpu'` | Install: `pip install wgpu` |
| GPU memory error | Reduce volume size or patch radius |
| Incorrect results | Check `tau_factor` and patch radius settings |
| Slow GPU performance | Verify GPU drivers updated; check workgroup_size |

---

## Future Improvements

1. **float64 GPU support** — Use separate shader for double precision
2. **Adaptive tau_factor** — Auto-tune based on noise statistics
3. **Boundary handling** — Options: reflection, periodic, replicate
4. **Streaming mode** — Process large volumes in slices
5. **CUDA backend** — PyCUDA or CuPy alternative to wgpu
6. **OpenMP parallelization** — Multi-threaded CPU backend

---

## License & Attribution

MPPCA implementation based on:
- Veraart et al. (2016) "Diffusion MRI noise mapping using random matrix theory"
- Local Patch Statistics Estimation (LPSE) framework

---

**Last Updated:** 2025
**Authors:** GPU/CPU MPPCA Implementation Team
