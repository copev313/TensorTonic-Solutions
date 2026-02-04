import numpy as np


def softmax(x):
    """Compute the softmax of input x. Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.

    Parameters
    ----------
    x: np.array
        One or two-dimensional array

    Returns
    -------
    np.array
        Softmax of the input (with same shape)
    """
    # NOTE: For 2D arrays, use axis=1, keepdims=True to maintain
    #   proper broadcasting dimensions.
    axis = 1 if x.ndim > 1 else 0

    # NOTE: Use broadcasting to subtract np.max(x, axis=...)
    #   before exponentiation for numerical stability.
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)

    # Sum exponentials:
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x
