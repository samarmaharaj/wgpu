import numpy as np


NUM_DIRS = 90
NUM_PARAMS = 22


def make_dki_design(num_dirs=NUM_DIRS, num_params=NUM_PARAMS, seed=123):
    """Create deterministic linearized DKI design matrix surrogate."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((num_dirs, num_params)).astype(np.float32)
    D[:, 0] = 1.0
    return D


def cpu_dki_fit(design, signal, eps=1e-6):
    """Linearized DKI fit via pseudo-inverse solve across all voxels."""
    design = np.asarray(design, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    if design.ndim != 2:
        raise ValueError("design must be 2D")
    if signal.ndim != 2:
        raise ValueError("signal must be 2D")
    if signal.shape[1] != design.shape[0]:
        raise ValueError("signal second dimension must match design rows")

    y = np.log(np.clip(signal, eps, None)).astype(np.float32)
    pinv = np.linalg.pinv(design).astype(np.float32)
    return y @ pinv.T


def run_cpu(num_voxels=100_000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    design = make_dki_design()
    return cpu_dki_fit(design, signal)
