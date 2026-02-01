import numpy as np

def manhattan_distance(x, y) -> float:
    """Compute the Manhattan (L1) distance between vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    absolute_diff = np.abs(x - y)
    return np.sum(absolute_diff)