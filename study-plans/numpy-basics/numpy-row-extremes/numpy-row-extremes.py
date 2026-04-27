import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    arr = np.array(data, dtype=np.float64)
    return np.stack((
        arr.max(axis=1),
        arr.argmax(axis=1),
        arr.min(axis=1),
        arr.argmin(axis=1),
    ))