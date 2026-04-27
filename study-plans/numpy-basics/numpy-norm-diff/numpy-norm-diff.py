import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a_clip = np.clip(a, lo, hi)
    b_clip = np.clip(b, lo, hi)
    a_num = a_clip - lo
    b_num = b_clip - lo
    denom = hi - lo
    a_out = np.divide(a_num, denom, out=np.zeros_like(a_num), where=denom != 0)
    b_out = np.divide(b_num, denom, out=np.zeros_like(b_num), where=denom != 0)
    return np.abs(a_out - b_out, dtype=np.float64)
    
    