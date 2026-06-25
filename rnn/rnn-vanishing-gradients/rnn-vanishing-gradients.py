import numpy as np


def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """Simulate gradient norm decay over T time steps.
    
    Returns list of gradient norms.
    """
    norms = []
    # Initial gradient norm:
    grad_norm = 1.0  
    # Compute l2 norm of W_hh:
    spec_norm = np.linalg.norm(W_hh, ord=2)

    for _ in range(T):
        norms.append(grad_norm)
        # Simulate gradient decay:
        grad_norm *= spec_norm

    return norms
