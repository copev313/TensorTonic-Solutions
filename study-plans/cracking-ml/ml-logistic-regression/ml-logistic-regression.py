import numpy as np


def logistic_regression(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.01,
    n_iters: int = 1000,
) -> tuple[np.ndarray, float]:
    """Logistic regression using gradient descent and
    binary cross-entropy loss function.

    Parameters:
    -----------
    X: list[list[float]]
        The input data, size (N, D).
    y: list[float]
        The target values, size (N,).
    lr: float, optional (default=0.01)
        The learning rate.
    n_iters: int, optional (default=1000)
        The number of epochs to train for.

    Returns:
    --------
    tuple (weights, bias)
    """
    # Convert to numpy:
    X_arr = np.array(X, dtype="float")
    y_arr = np.array(y, dtype="float")
    N_dim, D_dim = X_arr.shape
    # Initialize weights and bias:
    W = np.zeros(D_dim, dtype="float")
    B = 0.0

    # Training loop:
    for _ in range(n_iters):
        # 1. Compute linear output:
        z = X_arr @ W + B
        # 2. Apply sigmoid activation:
        sig = 1.0 / (1.0 + np.exp(-z))
        # 3. Compute gradients:
        c = 1.0 / N_dim
        diff = sig - y_arr
        dW = c * X_arr.T @ diff
        dB = c * np.sum(diff)
        # 4. Update weights and bias:
        W -= lr * dW
        B -= lr * dB

    return W, B