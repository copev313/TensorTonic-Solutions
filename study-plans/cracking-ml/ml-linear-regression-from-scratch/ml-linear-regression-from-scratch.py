import numpy as np


def linear_regression(
    X: list[list[float]], 
    y: list[float], 
    lr: float, 
    epochs: int,
) -> tuple[np.ndarray, float]:
    """Linear regression using gradient descent.

    Parameters:
    -----------
    X: list[list[float]]
        The input data, size (N, D).
    y: list[float]
        The target values, size (N,).
    lr: float
        The learning rate.
    epochs: int
        The number of epochs to train for.

    Returns:
    --------
    tuple (weights, bias)
    """
    # [CHECK] Validate input parameters:
    assert lr > 0, "Learning rate must be positive."
    assert epochs > 0, "Number of epochs must be positive."

    # Convert to numpy:
    X_arr = np.array(X, dtype="float")
    y_arr = np.array(y, dtype="float")
    N_dim, D_dim = X_arr.shape
    # Initialize weights and bias:
    W = np.zeros(D_dim, dtype="float")
    B = 0.0

    # Training loop:
    for _ in range(epochs):
        # 1. Forward pass - compute predictions:
        y_pred = X_arr @ W + B
        # 2. Compute loss:
        loss = (y_pred - y_arr)
        c = (2.0 / N_dim)
        # 3. Compute gradients to apply updates:
        dW = c * (X_arr.T @ loss)
        dB = c * np.sum(loss)
        # 4. Update weights and bias:
        W -= lr * dW
        B -= lr * dB

    # Round:
    return (np.round(W, 4), round(B, 4))