import numpy as np

def tanh(x):
    """Tanh activation function.
    
    Parameters:
    -----------
    x: int | float | list | np.ndarray
        Input value(s) to apply the tanh to.
    
    Returns:
    --------
    np.ndarray
        The tanh of the input value(s), with the same shape as the input.
    """
    x = np.asarray(x, dtype=float)
    x = np.atleast_1d(x)
    return np.tanh(x)