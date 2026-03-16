import numpy as np


def make_diagonal(v):
    """Creates a diagonal matrix provided a 1d vector of length n.

    Parameters:
    -----------
    v: array-like
        1d array of length n

    Returns:
    --------
    np.ndarray
        NumPy array of shape (n, n) with v on the main diagonal.
    """
    v_arr = np.array(v, dtype=float)
    return np.diag(v_arr)
