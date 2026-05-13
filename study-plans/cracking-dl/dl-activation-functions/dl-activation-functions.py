import numpy as np


def activation_functions(x, activation):
    """Computes both the activation output and its analytical derivative at the input.

    Returns: list [output, derivative]
    """
    x = np.array(x, dtype=np.float64)

    def result(x, dx):
        return [np.round(x, 4), np.round(dx, 4)]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    match activation:
        case "relu":
            x = np.maximum(0, x)
            dx = np.where(x > 0, 1.0, 0.0)
            return result(x, dx)
        case "leaky_relu":
            alpha = 0.01
            x = np.where(x > 0, x, alpha * x)
            dx = np.where(x > 0, 1.0, alpha)
            return result(x, dx)
        case "sigmoid":
            x = sigmoid(x)
            dx = x * (1 - x)
            return result(x, dx)
        case "tanh":
            x = np.tanh(x)
            dx = 1 - x**2
            return result(x, dx)
        case "gelu":
            k = 0.044715
            inner = np.sqrt(2 / np.pi) * (x + k * x**3)
            tanh = np.tanh(inner)
            output = 0.5 * x * (1 + tanh)
            sech2 = 1 - tanh**2
            d_inner = np.sqrt(2 / np.pi) * (1 + 3 * k * x**2)
            dx = 0.5 * (1 + tanh) + 0.5 * x * sech2 * d_inner
            return result(output, dx)
        case "swish":
            sig = sigmoid(x)
            output = x * sig
            dx = sig + x * sig * (1 - sig)
            return result(output, dx)
        case _:
            raise ValueError(f"Unknown activation function: {activation}")
