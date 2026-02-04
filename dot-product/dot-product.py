import numpy as np


def dot_product(x, y) -> float:
    """Compute the dot product of two 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sum(x * y)
