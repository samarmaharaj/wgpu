import numpy as np


def nlmeans_patch_weights_cpu(data, patch_radius=1, block_radius=3, sigma=1.0):
    """Reference CPU implementation for NLM patch-weight denoising."""
    X, Y, Z = data.shape
    P = patch_radius
    B = block_radius
    h2 = 2.0 * (sigma ** 2) * ((2 * P + 1) ** 3)

    padded = np.pad(data, P + B, mode="reflect").astype(np.float32)
    denoised = np.zeros_like(data, dtype=np.float32)

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                px, py, pz = x + P + B, y + P + B, z + P + B
                patch_i = padded[px - P:px + P + 1, py - P:py + P + 1, pz - P:pz + P + 1]

                weight_sum = 0.0
                value_sum = 0.0

                for dx in range(-B, B + 1):
                    for dy in range(-B, B + 1):
                        for dz in range(-B, B + 1):
                            nx, ny, nz = px + dx, py + dy, pz + dz
                            patch_j = padded[nx - P:nx + P + 1, ny - P:ny + P + 1, nz - P:nz + P + 1]

                            diff = patch_i - patch_j
                            dist2 = float(np.sum(diff * diff))
                            w = np.exp(-dist2 / h2)
                            weight_sum += w
                            value_sum += w * padded[nx, ny, nz]

                denoised[x, y, z] = value_sum / weight_sum

    return denoised
