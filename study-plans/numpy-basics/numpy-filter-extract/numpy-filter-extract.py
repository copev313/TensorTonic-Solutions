import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    arr = np.array(data, dtype = np.float64)
    r0, r1 = row_start, row_stop
    arr = arr[r0:r1, :].copy()
    filtered_idx = np.where(arr > threshold)
    return arr[filtered_idx].copy()