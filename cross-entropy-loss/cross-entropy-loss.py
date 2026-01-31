import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """Compute average cross-entropy loss for multi-class classification.
    
    Parameters
    ----------
    y_true: np.ndarray
        True class labels. Shape: (N,)
    y_pred: np.ndarray
        Predicted probabilities for each class. Shape: (N, C)
    
    Returns
    -------
    float
        Average CE loss over all samples.
    """
    y_hat = np.asarray(y_pred)
    y = np.asarray(y_true)
    length = y.shape[0]
    # Select predicted probs for true classes:
    select = y_hat[np.arange(length), y]
    # Calc. neg avg log likelihood:
    return -np.mean(np.log(select))