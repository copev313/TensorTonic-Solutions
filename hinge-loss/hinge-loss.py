import numpy as np


def hinge_loss(y_true, y_score, margin: float = 1.0, reduction: str = "mean") -> float:
    """Compute the hinge loss (binary SVM).

    Parameters
    ----------
    y_true: 1D array
        Array of {-1,+1}.
    y_score: 1D array
        Array of real scores, same shape as y_true.
    margin: float
        Margin parameter for hinge loss.
    reduction: str,
        Reduction method ("mean" or "sum").

    Returns
    -------
    float
        The computed hinge loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # [CHECK] Validate inputs:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must contain only -1 or +1.")
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    # Hinge loss:
    loss = np.maximum(0, margin - y_true * y_score)
    # Apply reduction:
    if reduction == "sum":
        return loss.sum()

    return loss.mean()
