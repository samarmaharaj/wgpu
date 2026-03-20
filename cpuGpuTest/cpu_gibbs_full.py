import time
from functools import partial
import multiprocessing as mp

import numpy as np
import scipy.fft

_fft = scipy.fft


def _determine_num_processes(num_processes):
    if num_processes is None:
        return 1
    if num_processes == 0:
        raise ValueError("num_processes cannot be 0")
    if num_processes < 0:
        cpu_count = mp.cpu_count()
        return max(1, cpu_count + num_processes + 1)
    return int(num_processes)


def _image_tv(x, *, axis=0, n_points=3):
    xs = x.copy() if axis else x.T.copy()
    xs = np.concatenate((xs[:, (-n_points - 1):], xs, xs[:, 0:(n_points + 1)]), axis=1)

    ptv = np.absolute(
        xs[:, (n_points + 1):(-n_points - 1)] - xs[:, (n_points + 2):(-n_points)]
    )
    ntv = np.absolute(
        xs[:, (n_points + 1):(-n_points - 1)] - xs[:, n_points:(-n_points - 2)]
    )
    for n in range(1, n_points):
        ptv = ptv + np.absolute(
            xs[:, (n_points + 1 + n):(-n_points - 1 + n)]
            - xs[:, (n_points + 2 + n):(-n_points + n)]
        )
        ntv = ntv + np.absolute(
            xs[:, (n_points + 1 - n):(-n_points - 1 - n)]
            - xs[:, (n_points - n):(-n_points - 2 - n)]
        )

    if axis:
        return ptv, ntv
    return ptv.T, ntv.T


def _gibbs_removal_1d(x, *, axis=0, n_points=3):
    dtype_float = np.promote_types(x.real.dtype, np.float32)
    ssamp = np.linspace(0.02, 0.9, num=45, dtype=dtype_float)
    xs = x.copy() if axis else x.T.copy()

    tvr, tvl = _image_tv(xs, axis=1, n_points=n_points)
    tvp = np.minimum(tvr, tvl)
    tvn = tvp.copy()

    isp = xs.copy()
    isn = xs.copy()
    sp = np.zeros(xs.shape, dtype=dtype_float)
    sn = np.zeros(xs.shape, dtype=dtype_float)
    N = xs.shape[1]
    c = _fft.fft(xs, axis=1)
    k = _fft.fftfreq(N, 1 / (2.0j * np.pi))
    k = k.astype(c.dtype, copy=False)

    for s in ssamp:
        ks = k * s
        img_p = abs(_fft.ifft(c * np.exp(ks), axis=1))
        tvsr, tvsl = _image_tv(img_p, axis=1, n_points=n_points)
        tvs_p = np.minimum(tvsr, tvsl)

        img_n = abs(_fft.ifft(c * np.exp(-ks), axis=1))
        tvsr, tvsl = _image_tv(img_n, axis=1, n_points=n_points)
        tvs_n = np.minimum(tvsr, tvsl)

        isp[tvp > tvs_p] = img_p[tvp > tvs_p]
        sp[tvp > tvs_p] = s
        tvp[tvp > tvs_p] = tvs_p[tvp > tvs_p]

        isn[tvn > tvs_n] = img_n[tvn > tvs_n]
        sn[tvn > tvs_n] = s
        tvn[tvn > tvs_n] = tvs_n[tvn > tvs_n]

    idx = np.nonzero(sp + sn)
    xs[idx] = (isp[idx] - isn[idx]) / (sp[idx] + sn[idx]) * sn[idx] + isn[idx]
    return xs if axis else xs.T


def _weights(shape):
    G0 = np.zeros(shape)
    G1 = np.zeros(shape)
    k0 = np.linspace(-np.pi, np.pi, num=shape[0])
    k1 = np.linspace(-np.pi, np.pi, num=shape[1])

    K1, K0 = np.meshgrid(k1[1:-1], k0[1:-1])
    cosk0 = 1.0 + np.cos(K0)
    cosk1 = 1.0 + np.cos(K1)
    G1[1:-1, 1:-1] = cosk0 / (cosk0 + cosk1)
    G0[1:-1, 1:-1] = cosk1 / (cosk0 + cosk1)

    G1[1:-1, 0] = G1[1:-1, -1] = 1
    G1[0, 0] = G1[-1, -1] = G1[0, -1] = G1[-1, 0] = 1 / 2
    G0[0, 1:-1] = G0[-1, 1:-1] = 1
    G0[0, 0] = G0[-1, -1] = G0[0, -1] = G0[-1, 0] = 1 / 2

    return G0, G1


def _gibbs_removal_2d(image, *, n_points=3, G0=None, G1=None):
    if G0 is None or G1 is None:
        G0, G1 = _weights(image.shape)

    img_c1 = _gibbs_removal_1d(image, axis=1, n_points=n_points)
    img_c0 = _gibbs_removal_1d(image, axis=0, n_points=n_points)

    C1 = _fft.fft2(img_c1)
    C0 = _fft.fft2(img_c0)
    imagec = abs(_fft.ifft2(_fft.fftshift(C1) * G1 + _fft.fftshift(C0) * G0))
    return imagec


def gibbs_removal(vol, *, slice_axis=2, n_points=3, inplace=True, num_processes=1):
    nd = vol.ndim
    if nd > 4:
        raise ValueError("Data have to be a 4D, 3D or 2D matrix")
    if nd < 2:
        raise ValueError("Data is not an image")
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a boolean.")

    num_processes = _determine_num_processes(num_processes)

    if slice_axis > 2:
        raise ValueError(
            "Different slices have to be organized along one of the 3 first matrix dimensions"
        )

    if nd == 3:
        vol = np.moveaxis(vol, slice_axis, 0)
    elif nd == 4:
        vol = np.moveaxis(vol, (slice_axis, 3), (0, 1))

    if nd == 4:
        inishap = vol.shape
        vol = vol.reshape((inishap[0] * inishap[1], inishap[2], inishap[3]))

    shap = vol.shape
    G0, G1 = _weights(shap[-2:])

    if not inplace:
        vol = vol.copy()

    if nd == 2:
        vol[:, :] = _gibbs_removal_2d(vol, n_points=n_points, G0=G0, G1=G1)
    else:
        if num_processes == 1:
            for i in range(shap[0]):
                vol[i, :, :] = _gibbs_removal_2d(vol[i, :, :], n_points=n_points, G0=G0, G1=G1)
        else:
            mp.set_start_method("spawn", force=True)
            pool = mp.Pool(num_processes)
            partial_func = partial(_gibbs_removal_2d, n_points=n_points, G0=G0, G1=G1)
            vol[:, :, :] = pool.map(partial_func, vol)
            pool.close()
            pool.join()

    if nd == 3:
        vol = np.moveaxis(vol, 0, slice_axis)
    if nd == 4:
        vol = vol.reshape(inishap)
        vol = np.moveaxis(vol, (0, 1), (slice_axis, 3))
    return vol


def gibbs_cpu(vol: np.ndarray) -> np.ndarray:
    """Full CPU Gibbs removal wrapper (DIPY reference implementation).

    Parameters
    ----------
    vol : ndarray
        4D volume with shape (X, Y, Z, N_gradients), float32 preferred.

    Returns
    -------
    ndarray
        Gibbs-corrected volume, float32.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim != 4:
        raise ValueError("vol must be 4D with shape (X, Y, Z, N_gradients)")
    out = gibbs_removal(vol, slice_axis=2, n_points=3, inplace=False, num_processes=1)
    return np.asarray(out, dtype=np.float32)


if __name__ == "__main__":
    sizes = [
        (32, 32, 32, 16),
        (64, 64, 32, 32),
        (80, 80, 80, 1),
    ]

    for shape in sizes:
        rng = np.random.default_rng(42)
        vol = (rng.standard_normal(shape) * 100 + 500).astype(np.float32)
        t0 = time.perf_counter()
        out = gibbs_cpu(vol)
        ms = (time.perf_counter() - t0) * 1000
        print(f"shape={shape} cpu_ms={ms:.1f} out_dtype={out.dtype}")
