import numpy as np


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        # Xavier initialization:
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(
            2.0 / (input_dim + hidden_dim)
        )
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(
            2.0 / (2 * hidden_dim)
        )
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + output_dim)
        )
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """Forward pass through entire sequence.

        Returns (y_seq, h_final).
        """
        N, T, _ = X.shape
        hidden_states = []

        if h_0 is None:
            h_0 = np.zeros((N, self.hidden_dim))

        # Iterate over all T time steps - applying RNN cell with tanh activation:
        for t in range(T):
            # Get current input at time step t:
            x_t = X[:, t, :]
            # Compute + collect new hidden state using RNN cell:
            h_t = np.tanh(x_t @ self.W_xh.T + h_0 @ self.W_hh.T + self.b_h)
            hidden_states.append(h_t)
            # Update h_0 for next time step:
            h_0 = h_t

        # Return tuple of (y_seq, h_final) - with shapes 
        # (batch_size, T, output_dim) and (batch_size, hidden_dim):
        y_seq = np.stack([h @ self.W_hy.T + self.b_y for h in hidden_states], axis=1)
        h_final = hidden_states[-1]
        return y_seq, h_final
