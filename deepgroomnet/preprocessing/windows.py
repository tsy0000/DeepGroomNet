import numpy as np


def window_starts(n_frames, window, stride):
    if n_frames < window:
        return np.array([0], dtype=int)
    starts = np.arange(0, n_frames - window + 1, int(stride), dtype=int)
    if starts[-1] != n_frames - window:
        starts = np.r_[starts, n_frames - window]
    return starts
