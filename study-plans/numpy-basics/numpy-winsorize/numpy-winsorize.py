import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    arr = np.array(data, dtype=np.float64)
    lo_q = np.percentile(arr, lo_q, axis=0)
    hi_q = np.percentile(arr, hi_q, axis=0)
    return np.stack((
        np.clip(arr, lo_q, hi_q),
        arr < lo_q,
        arr > hi_q,
    ))