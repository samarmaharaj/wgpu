import numpy as np


def mppca_proxy_cpu(data, patch_radius=1, tau=1.2, eps=1e-6):
    """MPPCA-inspired local variance shrinkage (CPU reference).

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, C)
        4D diffusion-like volume.
    patch_radius : int
        Spatial patch radius.
    tau : float
        Shrinkage strength.
    eps : float
        Numerical stability constant.
    """
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("data must be 4D (X, Y, Z, C)")

    x, y, z, channels = data.shape
    count = float((2 * patch_radius + 1) ** 3)
    calc_dtype = np.float64 if data.dtype == np.float64 else np.float32
    work = data.astype(calc_dtype, copy=False)
    out = np.empty_like(work, dtype=calc_dtype)

    def reflect_idx(i, n):
        idx = int(i)
        if idx < 0:
            idx = -idx - 1
        if idx >= n:
            idx = 2 * n - idx - 1
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        return idx

    for i in range(x):
        for j in range(y):
            for k in range(z):
                means = np.zeros(channels, dtype=calc_dtype)
                variances = np.zeros(channels, dtype=calc_dtype)

                for c in range(channels):
                    s1 = calc_dtype(0.0)
                    for dx in range(-patch_radius, patch_radius + 1):
                        xx = reflect_idx(i + dx, x)
                        for dy in range(-patch_radius, patch_radius + 1):
                            yy = reflect_idx(j + dy, y)
                            for dz in range(-patch_radius, patch_radius + 1):
                                zz = reflect_idx(k + dz, z)
                                s1 += work[xx, yy, zz, c]
                    mean_c = s1 / count
                    means[c] = mean_c

                    s2 = calc_dtype(0.0)
                    for dx in range(-patch_radius, patch_radius + 1):
                        xx = reflect_idx(i + dx, x)
                        for dy in range(-patch_radius, patch_radius + 1):
                            yy = reflect_idx(j + dy, y)
                            for dz in range(-patch_radius, patch_radius + 1):
                                zz = reflect_idx(k + dz, z)
                                d = work[xx, yy, zz, c] - mean_c
                                s2 += d * d
                    variances[c] = s2 / count

                sigma2 = np.mean(variances)
                center = work[i, j, k, :]
                shrink = np.maximum(calc_dtype(0.0), calc_dtype(1.0) - (tau * sigma2) / (variances + eps))
                out[i, j, k, :] = means + shrink * (center - means)

    return out.astype(calc_dtype, copy=False)