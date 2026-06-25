import numpy as np


def matrix_trace(A):
    """Compute the trace of a square matrix (sum of diagonal elements)."""
    A = np.asarray(A)
    assert A.shape[0] == A.shape[1], "Input must be a square matrix."
    # Edge case (1x1 matrix):
    if A.shape[0] == 1:
        return A.item(0, 0)

    return sum([A[i, i] for i in range(A.shape[0])])