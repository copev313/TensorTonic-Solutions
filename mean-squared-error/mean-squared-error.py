import numpy as np


def mean_squared_error(y_pred, y_true):
    """Compute Mean Squared Error (MSE) between predicted and true values.

    Parameters:
    -----------
    y_pred: array-like
        Predicted values (list, scalar, or np.ndarray).
    y_true: array-like
        True values (list, scalar, or np.ndarray).

    Returns:
    --------
    float or None
        MSE calculation. Returns None if input shapes do not match.
    """
    y_hat = np.array(y_pred)
    y = np.array(y_true)

    # [CASE] Shape mismatch:
    if y_hat.shape != y.shape:
        return None

    diff = (y_hat - y) ** 2
    return np.mean(diff)
