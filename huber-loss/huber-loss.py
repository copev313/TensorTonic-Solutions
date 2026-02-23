import numpy as np


def huber_loss(y_true, y_pred, delta: float = 1.0):
    """Compute Huber Loss for regression.

    Parameters
    ----------
    y_true: array-like
        True target values.
    y_pred: array-like
        Predicted values.
    delta: float, optional (default=1.0)
        Threshold parameter for controlling the transition between L1 and L2 loss.

    Returns
    -------
    float
        Huber loss value.
    """
    # [CHECKS] Validate input:
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    if delta <= 0:
        raise ValueError("Delta must be positive.")

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)
    loss = np.where(
        abs_error <= delta,
        # L2 loss for small errors:
        0.5 * abs_error**2,
        # L1 loss for large errors:
        delta * (abs_error - 0.5 * delta),
    )
    return np.mean(loss)
