import numpy as np


def _reflect_idx(i, n):
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


def _pca_classifier(evals_asc, nvoxels):
    vals = evals_asc
    if vals.size > nvoxels - 1:
        vals = vals[-(nvoxels - 1):]

    var = np.mean(vals)
    c = vals.size - 1
    r = vals[c] - vals[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var
    while r > 0 and c > 0:
        var = np.mean(vals[:c])
        c -= 1
        r = vals[c] - vals[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var
    return var, c + 1


def mppca_hybrid_cpu(data, patch_radius=1):
    """Hybrid-like MPPCA reference with explicit CPU eigh().

    Architecture:
    - patch extraction
    - mean + covariance
    - eigh(cov)
    - threshold eigenvectors
    - reconstruct + overlap accumulation
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError("data must be 4D (X, Y, Z, C)")

    x, y, z, channels = data.shape
    patch_side = 2 * patch_radius + 1
    patch_voxels = patch_side ** 3
    nvox = x * y * z

    means = np.zeros((nvox, channels), dtype=np.float32)
    covs = np.zeros((nvox, channels, channels), dtype=np.float32)

    def voxel_index(ix, iy, iz):
        return (ix * y + iy) * z + iz

    for ix in range(x):
        for iy in range(y):
            for iz in range(z):
                vid = voxel_index(ix, iy, iz)
                patch = np.zeros((patch_voxels, channels), dtype=np.float32)
                s = 0
                for dx in range(-patch_radius, patch_radius + 1):
                    px = _reflect_idx(ix + dx, x)
                    for dy in range(-patch_radius, patch_radius + 1):
                        py = _reflect_idx(iy + dy, y)
                        for dz in range(-patch_radius, patch_radius + 1):
                            pz = _reflect_idx(iz + dz, z)
                            patch[s, :] = data[px, py, pz, :]
                            s += 1

                m = patch.mean(axis=0)
                means[vid, :] = m
                centered = patch - m
                covs[vid, :, :] = (centered.T @ centered) / float(patch_voxels)

    projectors = np.zeros((nvox, channels, channels), dtype=np.float32)
    for vid in range(nvox):
        evals, evecs = np.linalg.eigh(covs[vid])
        _, ncomps = _pca_classifier(evals, patch_voxels)
        evecs[:, :ncomps] = 0.0
        projectors[vid, :, :] = (evecs @ evecs.T).astype(np.float32)

    out = np.zeros_like(data, dtype=np.float32)
    theta = np.zeros((x, y, z), dtype=np.float32)

    for ox in range(x):
        for oy in range(y):
            for oz in range(z):
                x_vec = data[ox, oy, oz, :]
                accum = np.zeros(channels, dtype=np.float32)
                count = 0.0

                for cx_d in range(-patch_radius, patch_radius + 1):
                    cx = _reflect_idx(ox + cx_d, x)
                    for cy_d in range(-patch_radius, patch_radius + 1):
                        cy = _reflect_idx(oy + cy_d, y)
                        for cz_d in range(-patch_radius, patch_radius + 1):
                            cz = _reflect_idx(oz + cz_d, z)
                            cvid = voxel_index(cx, cy, cz)
                            m = means[cvid]
                            b = projectors[cvid]
                            x_centered = x_vec - m
                            x_est = x_centered @ b + m
                            accum += x_est
                            count += 1.0

                out[ox, oy, oz, :] = accum / count
                theta[ox, oy, oz] = count

    return out, theta