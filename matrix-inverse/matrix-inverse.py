import numpy as np


def matrix_inverse(A):
    """Compute the inverse of a square, non-singular matrix A.

    Parameters:
    -----------
    A: np.ndarray
        A square, non-singular matrix of shape (n, n).

    Returns:
    --------
    np.ndarray
        A inverse of shape (n, n) such that A @ A_inv ≈ I.
        If A is singular, returns None.
    """
    # Validate input is 2D and square
    # Check if matrix is singular (return None if singular or invalid)
    # Use NumPy functions like np.linalg.inv()
    mat_A = np.array(A)
    # [CHECK] 2D and square:
    if mat_A.ndim != 2 or mat_A.shape[0] != mat_A.shape[1]:
        return None
    # [CHECK] Singular: Check if matrix is singular by computing np.linalg.det(A) and comparing with a small threshold like 1e-10.
    if abs(np.linalg.det(mat_A)) < 1e-10:
        return None

    # Compute inverse and verify:
    A_inv = np.linalg.inv(mat_A)
    mat_I = np.eye(mat_A.shape[0])
    if not np.allclose(mat_A @ A_inv, mat_I, atol=1e-7):
        return None

    return A_inv
