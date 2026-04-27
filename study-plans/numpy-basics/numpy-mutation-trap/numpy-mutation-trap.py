import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    arr = np.array(data, dtype=np.float64)
    orig_row = arr[row_idx].copy()
    clipped_row = np.clip(orig_row, lo, hi)
    return np.stack((orig_row, clipped_row))