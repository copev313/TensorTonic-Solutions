import numpy as np


def dropout(X, mask, drop_prob, mode) -> list[list[float]]:
    """Dropout with inverted scaling.
    
    Parameters:
    -----------
    X: list[list[float]]
        2D array of input values.
    mask: list[list[bool]]
        2D array of boolean values (same shape as X).
    drop_prob: float
        Probability of dropping a unit (between 0 and 1).
    mode: str
        'train' or 'test' mode.
    
    Returns:
    --------
    list[list[float]]
        2D array of output values after applying dropout
        (rounded to 4 decimal places).
    """
    X_arr, mask_arr = np.array(X, dtype="float"), np.array(mask, dtype="float")
    # [CHECK] Validate inputs:
    assert 0 <= drop_prob < 1, "drop_prob must be in the range [0, 1)"
    assert mode in ["train", "test"], "mode must be 'train' or 'test'"
    assert X_arr.shape == mask_arr.shape, "X and mask must have the same shape"

    # [CASE] Training mode:
    if mode == "train":
        p_inv = 1 - drop_prob
        # Apply element-wise mask & Scale to keep expected value:
        out = (X_arr * mask_arr) * (1 / p_inv)
        return np.round(out, 4).tolist()

    # [CASE] Test mode:
    return X_arr.tolist()
