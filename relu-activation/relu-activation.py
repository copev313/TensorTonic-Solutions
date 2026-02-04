import numpy as np


def relu(x):
    """ReLU activation function.

    Parameters
    ----------
    x: float | list | np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        ReLU values.
    """
    x = np.asarray(x, dtype=float)
    x = np.atleast_1d(x)
    return np.maximum(0, x)
