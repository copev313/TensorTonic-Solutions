import numpy as np


def minmax_scale(X, axis=0, eps=1e-12):
    """Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Array conversion:
    X = np.asarray(X)
    # Calc min/max per axis:
    minimum = np.min(X, axis=axis, keepdims=True)
    maximum = np.max(X, axis=axis, keepdims=True)
    denom = np.maximum(maximum - minimum, eps)
    return (X - minimum) / denom
    