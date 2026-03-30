import numpy as np


NUM_DIRS = 90
NUM_COEFFS = 7


def make_dti_design(num_dirs=NUM_DIRS, num_coeffs=NUM_COEFFS, seed=42):
    """Create a deterministic DTI design matrix surrogate for WLS testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_dirs, num_coeffs)).astype(np.float32)
    X[:, 0] = 1.0
    return X


def cpu_dti_wls(design, signal, eps=1e-6):
    """Voxelwise weighted least squares DTI fit.

    Parameters
    ----------
    design : ndarray, shape (num_dirs, num_coeffs)
    signal : ndarray, shape (num_voxels, num_dirs)
        Positive diffusion signal values.
    """
    design = np.asarray(design, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)

    if design.ndim != 2:
        raise ValueError("design must be 2D")
    if signal.ndim != 2:
        raise ValueError("signal must be 2D")
    if signal.shape[1] != design.shape[0]:
        raise ValueError("signal second dimension must match design rows")

    V, D = signal.shape
    C = design.shape[1]
    y = np.log(np.clip(signal, eps, None)).astype(np.float32)
    out = np.empty((V, C), dtype=np.float32)

    Xt = design.T.astype(np.float32)
    eye = np.eye(C, dtype=np.float32)

    for v in range(V):
        w = np.square(np.clip(signal[v], eps, None)).astype(np.float32)
        XtW = Xt * w[None, :]
        XtWX = XtW @ design
        XtWy = XtW @ y[v]
        out[v] = np.linalg.solve(XtWX + 1e-6 * eye, XtWy)

    return out


def run_cpu(num_voxels=100_000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    design = make_dti_design()
    return cpu_dti_wls(design, signal)
