import numpy as np


def dropout(x, p=0.5, rng=None):
    """Apply inverted dropout to input x with drop probability p.

    Return (output, dropout_pattern).
    """
    x = np.asarray(x)

    if not 0 <= p <= 1:
        raise ValueError("p must be in the range [0, 1]")

    if rng is None:
        rng = np.random.default_rng()

    keep_prob = 1.0 - p
    keep_mask = rng.random(x.shape) >= p

    if keep_prob == 0.0:
        dropout_pattern = np.zeros_like(x, dtype=x.dtype)
        output = np.zeros_like(x)
    else:
        dropout_pattern = keep_mask.astype(x.dtype, copy=False) / keep_prob
        output = x * dropout_pattern

    return output, dropout_pattern