import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity arrays.

    Parameters
    ----------
    a: np.ndarray
        1D array
    b: np.ndarray
        1D array of same length as a

    Returns:
    -------
    float
        Cosine similarity between a and b
    """
    # Dot product of a and b:
    dotted = np.dot(a, b)
    # Magnitudes of a and b:
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    # [CASE] Handle zero vectors:
    if mag_a == 0 or mag_b == 0:
        return 0

    return dotted / (mag_a * mag_b)