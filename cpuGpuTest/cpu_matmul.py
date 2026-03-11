import numpy as np


def cpu_matmul(a, b):
    return np.matmul(a, b)


def run_cpu(n=1024):
    rng = np.random.default_rng(0)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return cpu_matmul(a, b)
