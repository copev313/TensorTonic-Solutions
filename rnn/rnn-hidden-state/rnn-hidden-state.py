import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.
    """
    size = (batch_size, hidden_dim)
    return np.zeros(size)