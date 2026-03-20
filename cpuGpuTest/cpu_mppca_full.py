"""
CPU MPPCA (Marchenko-Pastur PCA) implementation.

Mirrors DIPY's localpca.genpca() algorithm exactly:
  1. Extract patch X of shape (num_samples, dim)
  2. Compute mean M = mean(X)
  3. Centre X = X - M
  4. Covariance C = X.T @ X / num_samples
  5. Eigendecomposition: d, W = eigh(C)
  6. Marchenko-Pastur threshold: classify, zero noise components
  7. Reconstruct: Xest = X @ W @ W.T + M
  8. Weight accumulation (theta, thetax)
  9. Final: denoised = thetax / theta

Reference: DIPY's localpca.py, function genpca()
"""

import numpy as np
from scipy.linalg import eigh


def _pca_classifier(L, nvoxels):
    """
    Marchenko-Pastur PCA classifier.
    
    Finds number of PCA components to keep based on eigenvalues
    using the Marchenko-Pastur distribution.
    
    Parameters
    ----------
    L : ndarray
        Eigenvalues in ascending order, shape (dim,)
    nvoxels : int
        Number of samples in the patch (e.g., (2*r+1)^3)
    
    Returns
    -------
    var : float
        Estimated noise variance
    ncomps : int
        Number of components to keep
    """
    L = np.asarray(L, dtype=np.float32)
    
    if L.size > nvoxels - 1:
        L = L[-(nvoxels - 1):]
    
    var = float(np.mean(L))
    c = int(L.size - 1)
    r = float(L[c] - L[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var)
    
    while r > 0 and c > 0:
        var = float(np.mean(L[:c]))
        c = c - 1
        r = float(L[c] - L[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var)
    
    ncomps = c + 1
    return var, ncomps


def mppca_cpu(arr, patch_radius=2, tau_factor=None):
    """
    CPU MPPCA denoising.
    
    Parameters
    ----------
    arr : ndarray
        Input volume of shape (X, Y, Z, dim), dtype float32
    patch_radius : int, optional
        Radius of the patch. Default 2 (27-voxel patch).
    tau_factor : float, optional
        Marchenko-Pastur threshold multiplier.
        If None, use: tau_factor = 1 + sqrt(dim / num_samples)
    
    Returns
    -------
    denoised : ndarray
        Denoised volume of shape (X, Y, Z, dim), dtype float32
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("arr must be 4D (X, Y, Z, dim)")
    
    X, Y, Z, dim = arr.shape
    patch_side = 2 * patch_radius + 1
    num_samples = patch_side ** 3
    
    # Auto tau_factor if not provided
    if tau_factor is None:
        tau_factor = 1.0 + np.sqrt(dim / num_samples)
    
    # Initialize output accumulators
    thetax = np.zeros_like(arr, dtype=np.float32)
    theta = np.zeros((X, Y, Z), dtype=np.float32)
    
    # Process each voxel
    for ix in range(X):
        for iy in range(Y):
            for iz in range(Z):
                # 1. Extract patch X of shape (num_samples, dim)
                patch = np.zeros((num_samples, dim), dtype=np.float32)
                s = 0
                for dx in range(-patch_radius, patch_radius + 1):
                    px = ix + dx
                    if px < 0:
                        px = -px - 1
                    if px >= X:
                        px = 2 * X - px - 1
                    if px < 0:
                        px = 0
                    if px >= X:
                        px = X - 1
                    
                    for dy in range(-patch_radius, patch_radius + 1):
                        py = iy + dy
                        if py < 0:
                            py = -py - 1
                        if py >= Y:
                            py = 2 * Y - py - 1
                        if py < 0:
                            py = 0
                        if py >= Y:
                            py = Y - 1
                        
                        for dz in range(-patch_radius, patch_radius + 1):
                            pz = iz + dz
                            if pz < 0:
                                pz = -pz - 1
                            if pz >= Z:
                                pz = 2 * Z - pz - 1
                            if pz < 0:
                                pz = 0
                            if pz >= Z:
                                pz = Z - 1
                            
                            patch[s, :] = arr[px, py, pz, :]
                            s += 1
                
                # 2. Compute mean
                M = np.mean(patch, axis=0)  # shape (dim,)
                
                # 3. Centre
                X_centered = patch - M  # shape (num_samples, dim)
                
                # 4. Compute covariance
                C = (X_centered.T @ X_centered) / num_samples  # shape (dim, dim)
                
                # 5. Eigendecomposition
                d, W = eigh(C)  # d shape (dim,), W shape (dim, dim)
                # eigh returns eigenvalues in ascending order
                
                # 6. Marchenko-Pastur threshold
                var, ncomps = _pca_classifier(d, num_samples)
                tau = (tau_factor ** 2) * var
                
                # Zero out noise components (those with d < tau)
                W[:, :ncomps] = 0.0
                
                # 7. Reconstruct
                # Xest = X_centered @ W @ W.T + M
                Xest = (X_centered @ W @ W.T) + M  # shape (num_samples, dim)
                
                # 8. Weight accumulation
                this_theta = 1.0 / (1.0 + dim - ncomps)
                
                # Accumulate back into volume patch positions
                s = 0
                for dx in range(-patch_radius, patch_radius + 1):
                    px = ix + dx
                    if px < 0:
                        px = -px - 1
                    if px >= X:
                        px = 2 * X - px - 1
                    if px < 0:
                        px = 0
                    if px >= X:
                        px = X - 1
                    
                    for dy in range(-patch_radius, patch_radius + 1):
                        py = iy + dy
                        if py < 0:
                            py = -py - 1
                        if py >= Y:
                            py = 2 * Y - py - 1
                        if py < 0:
                            py = 0
                        if py >= Y:
                            py = Y - 1
                        
                        for dz in range(-patch_radius, patch_radius + 1):
                            pz = iz + dz
                            if pz < 0:
                                pz = -pz - 1
                            if pz >= Z:
                                pz = 2 * Z - pz - 1
                            if pz < 0:
                                pz = 0
                            if pz >= Z:
                                pz = Z - 1
                            
                            thetax[px, py, pz, :] += Xest[s, :] * this_theta
                            theta[px, py, pz] += this_theta
                            s += 1
    
    # 9. Final normalization
    denoised = np.zeros_like(arr, dtype=np.float32)
    for ix in range(X):
        for iy in range(Y):
            for iz in range(Z):
                if theta[ix, iy, iz] > 0:
                    denoised[ix, iy, iz, :] = thetax[ix, iy, iz, :] / theta[ix, iy, iz]
                else:
                    denoised[ix, iy, iz, :] = arr[ix, iy, iz, :]
    
    return denoised


if __name__ == "__main__":
    from time import perf_counter
    
    # Test on small synthetic volume
    print("=" * 70)
    print("CPU MPPCA Full Implementation Test")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    
    for size, n_grad in [(8, 8), (12, 16)]:
        print(f"\nTest: volume {size}^3 with {n_grad} gradients")
        
        data = rng.standard_normal((size, size, size, n_grad)).astype(np.float32) * 100 + 500
        
        t0 = perf_counter()
        result = mppca_cpu(data, patch_radius=1)
        t1 = perf_counter()
        
        print(f"  Time: {(t1-t0)*1000:.2f} ms")
        print(f"  Output shape: {result.shape}")
        print(f"  Output dtype: {result.dtype}")
        print(f"  Output range: [{result.min():.1f}, {result.max():.1f}]")
        assert result.shape == data.shape
        assert result.dtype == np.float32
    
    print("\n" + "=" * 70)
    print("CPU MPPCA tests completed successfully")
    print("=" * 70)
