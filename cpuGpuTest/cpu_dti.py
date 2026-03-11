import numpy as np

# DTI standard: 90 gradient directions → 7 tensor coefficients (6 unique + S0)
NUM_DIRS = 90
NUM_COEFFS = 7


def make_w_inv(num_dirs=NUM_DIRS, num_coeffs=NUM_COEFFS, seed=42):
    """
    Simulate the pre-computed OLS pseudo-inverse design matrix:
        W_inv = (W^T W)^{-1} W^T   shape: (num_coeffs, num_dirs)
    In real DTI this encodes b-values and gradient directions.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((num_dirs, num_coeffs)).astype(np.float32)
    return np.linalg.pinv(W).astype(np.float32)  # (num_coeffs, num_dirs)


def cpu_dti_ols(w_inv, signal):
    """
    OLS DTI fit across all voxels via a single batched matmul.

    Math: X = W_inv @ ln(S)  for every voxel simultaneously.

    w_inv:  (num_coeffs, num_dirs)   - pre-computed pseudo-inverse
    signal: (num_voxels, num_dirs)   - diffusion signal (must be positive)
    returns (num_voxels, num_coeffs) - tensor coefficients per voxel
    """
    return np.log(signal) @ w_inv.T


def run_cpu(num_voxels=1_000_000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    w_inv = make_w_inv()
    return cpu_dti_ols(w_inv, signal)
