import numpy as np


def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """Performs one AdaGrad update step.
    
    Parameters:
    -----------
    w : np.ndarray
        Current parameters
    g : np.ndarray
        Current gradients (same shape as w)
    G : np.ndarray
        Accumulated squared gradients (same shape as w)
    lr : float, optional (default=0.01)
        Learning rate
    eps : float, optional (default=1e-8)
        Small constant for numerical stability
    
    Returns:
    --------
    w_new : np.ndarray
        Updated parameters after the AdaGrad step
    G_new : np.ndarray
        Updated accumulated squared gradients after the AdaGrad step
    """
    # Array conversion:
    w = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)
    # 1. Accumulate squared gradients:
    G_new = G + g**2
    # 2. Parameter update:
    w_new = w - (lr / (np.sqrt(G_new + eps))) * g
    return w_new, G_new
