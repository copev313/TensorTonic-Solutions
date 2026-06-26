import numpy as np


def pearson_correlation(X):
    """Compute Pearson correlation matrix from dataset X.

    Parameters
    ----------
    X: list[list[float]] | np.ndarray
        Dataset with shape (N, D)

    Returns
    -------
    np.ndarray
        Pearson correlation matrix with shape (D, D)
    """
    X = np.asarray(X)
    sd = np.std(X, axis=0, ddof=1)
    cov = np.cov(X, rowvar=False)
    
    zero_sd_mask = (sd == 0)
    sd[zero_sd_mask] = 1.0
    mat = cov / np.outer(sd, sd)
    mat[zero_sd_mask, :] = "nan"
    mat[:, zero_sd_mask] = "nan"
    return mat