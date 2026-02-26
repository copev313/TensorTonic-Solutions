import numpy as np
import math


def gelu(x):
    """Compute the Gaussian Error Linear Unit (exact version using erf).

    Parameters
    ----------
    x: list or np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output array after applying the GELU activation function.
    """
    x_arr = np.array(x, dtype=float)
    # Exact formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Use np.vectorize(math.erf) to make the error function work with arrays.
    erf_func = np.vectorize(math.erf)
    return 0.5 * x_arr * (1 + erf_func(x_arr / math.sqrt(2)))
