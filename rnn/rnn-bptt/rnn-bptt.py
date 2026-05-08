import numpy as np


def bptt_single_step(
    dh_next: np.ndarray,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W_hh: np.ndarray,
) -> tuple:
    """Backprop through one RNN time step.

    Returns (dh_prev, dW_hh).
    """
    # Compute tanh derivative:
    dtanh = (1 - h_t**2) * dh_next
    # Compute weight gradients:
    dW_hh = np.dot(dtanh.T, h_prev)
    # Compute hidden state gradient:
    dh_prev = np.dot(dtanh, W_hh)
    return dh_prev, dW_hh
