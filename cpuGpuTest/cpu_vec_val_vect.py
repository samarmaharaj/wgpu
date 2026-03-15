import numpy as np


def cpu_vec_val_vect(evecs, evals):
    """CPU prototype for vec_val_vect: evecs @ diag(evals) @ evecs.T per sample."""
    evecs = np.asarray(evecs, dtype=np.float32)
    evals = np.asarray(evals, dtype=np.float32)

    if evecs.shape[-2:] != (3, 3):
        raise ValueError("Expected evecs shape (..., 3, 3)")
    if evals.shape != evecs.shape[:-2] + (3,):
        raise ValueError("Expected evals shape (..., 3) matching evecs batch shape")

    return np.einsum("...ik,...k,...jk->...ij", evecs, evals, evecs, optimize=True).astype(np.float32)
