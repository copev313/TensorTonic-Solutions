import numpy as np

def euclidean_distance(x, y):
    """Compute the Euclidean (L2) distance between vectors x and y.
    Returns result as a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.sum((x - y) ** 2)
    return np.sqrt(inside)