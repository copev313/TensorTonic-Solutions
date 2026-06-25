import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    Z = X @ W
    l2_norm = np.linalg.norm(Z, axis=1)
    gated = l2_norm >= threshold
    return Z * gated[:, np.newaxis]