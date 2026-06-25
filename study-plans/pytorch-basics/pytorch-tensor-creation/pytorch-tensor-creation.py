import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    match method:
        case "zeros":
            return torch.zeros(shape).tolist()
        case "ones":
            return torch.ones(shape).tolist()
        case _:
            return torch.full(shape, fill_value=value).tolist()