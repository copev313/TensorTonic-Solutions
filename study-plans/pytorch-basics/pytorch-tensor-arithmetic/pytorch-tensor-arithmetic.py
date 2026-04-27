import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    func = None
    match op:
        case "add":
            func = torch.add
        case "multiply":
            func = torch.mul
        case "matmul":
            func = torch.matmul
        case "power":
            func = torch.pow
        case "max":
            func = torch.max

    return func(x, y).tolist()