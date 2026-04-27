import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    arr = np.array(data, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / std