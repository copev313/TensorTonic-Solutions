import numpy as np


def covariance_matrix(X):
    """Compute covariance matrix from dataset X.

    Parameters
    ----------
    X: list[list[float]] | np.ndarray
        Dataset with shape (N, D)

    Returns
    -------
    np.ndarray | None
        Covariance matrix with shape (D, D). None for invalid input (N < 2 or not 2D)
    """
    X = np.asarray(X, dtype=float)
    # [CASE] Invalid input dims:
    if X.ndim != 2:
        return None

    N = X.shape[0]
    if N < 2:
        return None

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov_mat = (X_centered.T @ X_centered) / (N - 1)
    return cov_mat
