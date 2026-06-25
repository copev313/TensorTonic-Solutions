import numpy as np


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """A single Adam optimizer update step.
    
    Returns:
    --------
    param_new : np.ndarray
        Updated parameters after the Adam step.
    m_new : np.ndarray
        Updated first moment vector.
    v_new : np.ndarray
        Updated second moment vector.
    """
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    # 1. Update first moment:
    m_new = beta1 * m + (1 - beta1) * grad
    # 2. Update second moment:
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    # 3. Compute bias-correction:
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    # 4. Update parameters:
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param_new, m_new, v_new
