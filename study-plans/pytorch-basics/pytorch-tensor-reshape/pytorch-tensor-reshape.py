import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)
    match op:
        case "flatten":
            return torch.flatten(x, start_dim=0)
        case "squeeze":
            return x.view(-1).squeeze().tolist()
        case "transpose":
            return x.T.tolist()