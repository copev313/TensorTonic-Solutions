import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        l1_out = self.l1(x)
        relu_out = self.relu(l1_out)
        return self.l2(relu_out)
