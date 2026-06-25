import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    arr = np.array(data, dtype=np.float64)
    rounded = np.round(arr, decimals=decimals)
    floored = np.floor(arr)
    ceiling = np.ceil(arr)
    return np.stack([
        np.pad(x, pad_width, mode="constant", constant_values=0)
        for x in [rounded, floored, ceiling]
    ])