import numpy as np


def rnn_forward(
    X: np.ndarray, 
    h_0: np.ndarray, 
    W_xh: np.ndarray, 
    W_hh: np.ndarray, 
    b_h: np.ndarray,
) -> tuple:
    """Forward pass through entire sequence."""
    T = X.shape[1]
    hidden_states = []

    # Loop over T times steps of input tensor X with shape (batch, T, input_dim):
    for t in range(T):
        x_t = X[:, t, :]
        
        # Update hidden state using RNN cell formula with tanh activation:
        h = np.tanh(x_t @ W_xh.T + h_0 @ W_hh.T + b_h)
        hidden_states.append(h)
        # Update hidden state for next time step:
        h_0 = h  
    
    # Stack hidden states along the time axis - output shape (batch, T, hidden_dim):
    hidden_states = np.stack(hidden_states, axis=1)
    # Return tuple of (hidden_states, h_final) where the last hidden state has shape (batch, hidden_dim):
    return hidden_states, h_0
