import numpy as np


def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X, y = np.array(X), np.array(y)
    N, num_features = X.shape
    W = np.zeros(num_features)
    bias = 0.0

    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(W, x_i) + bias
            y_predicted = 1 if linear_output >= 0 else 0

            update = lr * (y[idx] - y_predicted)
            W += update * x_i
            bias += update

    return W.tolist(), bias
