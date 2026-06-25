import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.array(data, dtype=np.float64)
    dims = arr.shape
    match operation:
        case "flatten":
            return arr.flatten()
        case "transpose":
            return arr.T
        case "add_batch":
            return arr.reshape(1, *dims) 