import numpy as np


def rnn_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # Compute linear transform of prev. hidden state:
    hh_term = np.dot(h_prev, W_hh.T)
    # Compute linear transform of current input:
    input_h_term = np.dot(x_t, W_xh.T)
    # Add terms together with bias:
    combined = hh_term + input_h_term + b_h
    # Apply tanh activation function:
    return np.tanh(combined)
