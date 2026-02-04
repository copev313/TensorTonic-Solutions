import numpy as np


def sigmoid(x):
    """Vectorized sigmoid function.

    Parameters
    ----------
    x: int | float | list | np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Sigmoid values.
    """
    x = np.asarray(x, dtype=float)
    exp = np.exp(-x)
    return 1 / (1 + exp)
