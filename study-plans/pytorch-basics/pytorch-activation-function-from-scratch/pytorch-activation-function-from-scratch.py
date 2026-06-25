import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    match method:
        case "relu":
            return torch.clamp(x, min=0).tolist()
        case "sigmoid":
            return torch.div(1, (1+ torch.exp(-x))).tolist()
        case "leaky_relu":
            return torch.where(x > 0, x, 0.01 * x).tolist()
        case "tanh":
            return torch.div((torch.exp(x) - torch.exp(-x)), (torch.exp(x) + torch.exp(-x))).tolist()