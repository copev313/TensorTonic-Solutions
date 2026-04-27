import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return (
            x @ self.weight.t() + self.bias
        )
