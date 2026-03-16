import numpy as np


def expected_value_discrete(x, p):
    """Computes the expected value of a discrete random variable.

    Parameters:
    -----------
    x: array-like
        Array of possible values with shape (N,).
    p: array-like
        Array of corresponding probabilities with shape (N,).

    Returns:
    --------
    float
        Expected value
    """
    x_arr = np.asarray(x)
    p_arr = np.asarray(p)

    # [CHECK] Shapes must match:
    if x_arr.shape != p_arr.shape:
        raise ValueError("Shapes of x and p must match.")

    # [CHECK] Probabilities must sum to 1:
    if not np.isclose(np.sum(p_arr), 1, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.")
    
    # Compute expected value:
    return np.sum(x_arr * p_arr)
