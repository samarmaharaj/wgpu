import numpy as np


def _prepare_lengths(points):
    diffs = points[:, 1:, :] - points[:, :-1, :]
    seg_lens = np.linalg.norm(diffs, axis=-1)
    cumlen = np.concatenate(
        [np.zeros((points.shape[0], 1), dtype=np.float32), np.cumsum(seg_lens, axis=1, dtype=np.float32)],
        axis=1,
    )
    return cumlen


def cpu_set_number_of_points(streamlines, nb_points=50):
    """CPU prototype for streamline resampling to a fixed number of points."""
    streamlines = np.asarray(streamlines, dtype=np.float32)
    if streamlines.ndim != 3 or streamlines.shape[-1] != 3:
        raise ValueError("Expected streamlines shape (n_streamlines, n_in_points, 3)")

    cumlen = _prepare_lengths(streamlines)
    t_norm = np.linspace(0.0, 1.0, nb_points, dtype=np.float32)
    return cpu_set_number_of_points_precomputed(streamlines, cumlen, t_norm)


def cpu_set_number_of_points_precomputed(streamlines, cumlen, t_norm):
    n_streamlines, n_in, _ = streamlines.shape
    n_out = t_norm.shape[0]
    out = np.empty((n_streamlines, n_out, 3), dtype=np.float32)

    for s in range(n_streamlines):
        total = float(cumlen[s, -1])
        if total <= 0.0:
            out[s] = streamlines[s, 0]
            continue

        for j in range(n_out):
            target = float(t_norm[j]) * total
            seg = int(np.searchsorted(cumlen[s], target, side="right") - 1)
            if seg < 0:
                seg = 0
            if seg >= n_in - 1:
                seg = n_in - 2

            l0 = float(cumlen[s, seg])
            l1 = float(cumlen[s, seg + 1])
            denom = l1 - l0
            alpha = 0.0 if denom <= 1e-12 else (target - l0) / denom

            out[s, j] = (1.0 - alpha) * streamlines[s, seg] + alpha * streamlines[s, seg + 1]

    return out
