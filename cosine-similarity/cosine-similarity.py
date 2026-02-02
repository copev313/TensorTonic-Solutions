import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two Numpy arrays."""
    # Magnitudes of a and b:
    mag_prod = np.linalg.norm(a) * np.linalg.norm(b)
    return (np.dot(a, b) / mag_prod) if mag_prod != 0.0 else 0