import numpy as np


def matrix_transpose(A):
    """Compute the transpose of a matrix A.

    Parameters
    ----------
    A: np.array
        2d numpy array with shape (N, M)
    """    
    A = np.asarray(A)
    n, m = A.shape
    new_matrix = np.zeros((m, n))
    # Iterate over rows + cols:
    for i in np.arange(n):
        for j in np.arange(m):
            new_matrix[j, i] = A[i, j]

    return new_matrix
