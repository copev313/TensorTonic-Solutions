import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        # [CASE] Training mode -> Apply dropout:
        if self.training:
            if self.p == 1.0:
                return torch.zeros_like(x)

            dropout_mask = (torch.rand_like(x) > self.p).float()
            # Scale the output to maintain the expected value:
            return x * dropout_mask / (1.0 - self.p)

        # [CASE] Evaluation mode -> No dropout:
        return x

