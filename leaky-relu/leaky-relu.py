import numpy as np

def leaky_relu(x, alpha: float = 0.01):
    """Vectorized Leaky ReLU.

    Parameters
    ----------
    x: float | list | np.ndarray
        Input values.
    alpha: float
        Slope of the negative part.
    
    Returns
    -------
    np.ndarray
        Leaky ReLU values.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)