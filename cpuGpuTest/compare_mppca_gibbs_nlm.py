import argparse
from statistics import mean
from time import perf_counter

import numpy as np

from cpu_gibbs import gibbs_suppress_cpu
from cpu_mppca import mppca_proxy_cpu
from cpu_nlmeans import nlmeans_patch_weights_cpu
from gpu_gibbs import GpuGibbsSuppress
from gpu_mppca import GpuMPPCAProxy
from gpu_nlmeans import GpuNLMeans


def _time_func(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return mean(times)


def _error_stats(reference64, arr32, atol):
    err = np.abs(arr32.astype(np.float64) - reference64.astype(np.float64))
    return {
        "max_abs_diff": float(err.max()),
        "mean_abs_diff": float(err.mean()),
        "count_exceed_atol": int(np.count_nonzero(err > atol)),
        "allclose": bool(np.allclose(arr32, reference64, atol=atol, rtol=0.0)),
    }


def _fmt_ms(seconds):
    return f"{seconds * 1000:.3f} ms"


def _nlmeans_reference64(data, patch_radius=1, block_radius=3, sigma=50.0):
    x, y, z = data.shape
    p = patch_radius
    b = block_radius
    h2 = 2.0 * (sigma ** 2) * ((2 * p + 1) ** 3)

    padded = np.pad(data, p + b, mode="reflect").astype(np.float64)
    denoised = np.zeros_like(data, dtype=np.float64)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                pi, pj, pk = i + p + b, j + p + b, k + p + b
                patch_i = padded[pi - p:pi + p + 1, pj - p:pj + p + 1, pk - p:pk + p + 1]

                weight_sum = 0.0
                value_sum = 0.0

                for dx in range(-b, b + 1):
                    for dy in range(-b, b + 1):
                        for dz in range(-b, b + 1):
                            nx, ny, nz = pi + dx, pj + dy, pk + dz
                            patch_j = padded[nx - p:nx + p + 1, ny - p:ny + p + 1, nz - p:nz + p + 1]

                            diff = patch_i - patch_j
                            dist2 = float(np.sum(diff * diff))
                            w = np.exp(-dist2 / h2)
                            weight_sum += w
                            value_sum += w * padded[nx, ny, nz]

                denoised[i, j, k] = value_sum / weight_sum

    return denoised


def run(repeats=3):
    rng = np.random.default_rng(42)

    vol_nlm = (500.0 + 50.0 * rng.standard_normal((20, 20, 20))).astype(np.float32)
    vol_mppca = (800.0 + 60.0 * rng.standard_normal((16, 16, 16, 16))).astype(np.float32)
    vol_gibbs = (1000.0 + 80.0 * rng.standard_normal((56, 56, 56))).astype(np.float32)

    nlm_gpu = GpuNLMeans()
    mppca_gpu = GpuMPPCAProxy()
    gibbs_gpu = GpuGibbsSuppress()

    cases = [
        {
            "name": "NLM",
            "cpu64": lambda: _nlmeans_reference64(
                vol_nlm.astype(np.float64), patch_radius=1, block_radius=3, sigma=50.0
            ),
            "cpu32": lambda: nlmeans_patch_weights_cpu(
                vol_nlm.astype(np.float32), patch_radius=1, block_radius=3, sigma=50.0
            ),
            "gpu32": lambda: nlm_gpu.fit(
                vol_nlm.astype(np.float32), patch_radius=1, block_radius=3, sigma=50.0
            ),
            "tol": 1e-3,
        },
        {
            "name": "MPPCA",
            "cpu64": lambda: mppca_proxy_cpu(
                vol_mppca.astype(np.float64), patch_radius=1, tau=1.2
            ),
            "cpu32": lambda: mppca_proxy_cpu(
                vol_mppca.astype(np.float32), patch_radius=1, tau=1.2
            ),
            "gpu32": lambda: mppca_gpu.fit(
                vol_mppca.astype(np.float32), patch_radius=1, tau=1.2
            ),
            "tol": 2e-3,
        },
        {
            "name": "GIBBS",
            "cpu64": lambda: gibbs_suppress_cpu(vol_gibbs.astype(np.float64), alpha=0.8),
            "cpu32": lambda: gibbs_suppress_cpu(vol_gibbs.astype(np.float32), alpha=0.8),
            "gpu32": lambda: gibbs_gpu.fit(vol_gibbs.astype(np.float32), alpha=0.8),
            "tol": 1e-4,
        },
    ]

    results = []
    for case in cases:
        ref64 = case["cpu64"]()
        cpu32_out = case["cpu32"]()
        gpu32_out = case["gpu32"]()

        cpu_t = _time_func(case["cpu32"], repeats)
        gpu_t = _time_func(case["gpu32"], repeats)

        cpu_err = _error_stats(ref64, cpu32_out, case["tol"])
        gpu_err = _error_stats(ref64, gpu32_out, case["tol"])

        results.append(
            {
                "name": case["name"],
                "cpu_t": cpu_t,
                "gpu_t": gpu_t,
                "speedup": cpu_t / gpu_t if gpu_t > 0 else float("inf"),
                "tol": case["tol"],
                "cpu_err": cpu_err,
                "gpu_err": gpu_err,
            }
        )

    print("=== Runtime (mean over repeats) ===")
    for r in results:
        print(
            f"{r['name']}: CPU={_fmt_ms(r['cpu_t'])}, GPU={_fmt_ms(r['gpu_t'])}, "
            f"speedup={r['speedup']:.2f}x"
        )

    print("\n=== Float32 error vs Float64 reference ===")
    for r in results:
        print(f"{r['name']} (atol={r['tol']}):")
        print(
            "  CPU32 -> "
            f"max_abs={r['cpu_err']['max_abs_diff']:.3e}, "
            f"mean_abs={r['cpu_err']['mean_abs_diff']:.3e}, "
            f"count_exceed={r['cpu_err']['count_exceed_atol']}, "
            f"allclose={r['cpu_err']['allclose']}"
        )
        print(
            "  GPU32 -> "
            f"max_abs={r['gpu_err']['max_abs_diff']:.3e}, "
            f"mean_abs={r['gpu_err']['mean_abs_diff']:.3e}, "
            f"count_exceed={r['gpu_err']['count_exceed_atol']}, "
            f"allclose={r['gpu_err']['allclose']}"
        )

    by_gpu_time = sorted(results, key=lambda item: item["gpu_t"])
    print("\n=== Fastest by GPU runtime (lower is better) ===")
    for idx, r in enumerate(by_gpu_time, start=1):
        print(f"{idx}. {r['name']} -> {_fmt_ms(r['gpu_t'])}")

    by_speedup = sorted(results, key=lambda item: item["speedup"], reverse=True)
    print("\n=== Best speedup (CPU/GPU) ===")
    for idx, r in enumerate(by_speedup, start=1):
        print(f"{idx}. {r['name']} -> {r['speedup']:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MPPCA, GIBBS, and NLM on fixed test data with CPU/GPU timing and float32 error stats."
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats for CPU/GPU execution")
    args = parser.parse_args()

    run(repeats=args.repeats)


if __name__ == "__main__":
    main()