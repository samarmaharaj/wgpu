import numpy as np


def gibbs_suppress_cpu(data, alpha=0.8, eps=1e-6):
    """Gibbs-ringing suppression proxy based on local oscillation damping.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        3D volume.
    alpha : float
        Oscillation damping strength.
    eps : float
        Numerical stability constant.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("data must be 3D (X, Y, Z)")

    x, y, z = data.shape
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
                xm_i = reflect_idx(i - 1, x)
                xp_i = reflect_idx(i + 1, x)
                ym_i = reflect_idx(j - 1, y)
                yp_i = reflect_idx(j + 1, y)
                zm_i = reflect_idx(k - 1, z)
                zp_i = reflect_idx(k + 1, z)

                c = work[i, j, k]

                xm = work[xm_i, j, k]
                xp = work[xp_i, j, k]
                ym = work[i, ym_i, k]
                yp = work[i, yp_i, k]
                zm = work[i, j, zm_i]
                zp = work[i, j, zp_i]

                lap = (xm + xp + ym + yp + zm + zp) - 6.0 * c
                ratio = abs(lap) / (abs(c) + eps)
                w = min(1.0, max(0.0, alpha * ratio))
                smooth = (xm + xp + ym + yp + zm + zp) / 6.0
                out[i, j, k] = (1.0 - w) * c + w * smooth

    return out.astype(calc_dtype, copy=False)