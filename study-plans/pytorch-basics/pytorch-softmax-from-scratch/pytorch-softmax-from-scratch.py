import torch


def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    # Subtract row max:
    row_maxes = torch.max(logits, dim=1, keepdim=True).values
    shifted = logits - row_maxes
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=1, keepdim=True)
