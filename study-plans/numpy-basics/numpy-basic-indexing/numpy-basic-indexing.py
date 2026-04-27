import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.array(arr, dtype=np.float64)
    r0, r1 =  row_start, row_stop
    c0, c1 = col_start, col_stop
    return arr[r0:r1, c0:c1].copy()
