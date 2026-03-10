import numpy as np


def cpu_vector_add(a, b):
    return np.add(a, b, dtype=np.float32)


def run_cpu(n=10_000_000):
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)

    result = cpu_vector_add(a, b)

    return result