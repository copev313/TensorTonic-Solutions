import torch
import torch.nn as nn


def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    losses = []
    # Training step:
    for batch in dataloader:
        inputs, targets = batch
        # Clear gradients:
        optimizer.zero_grad()
        # 1. Forward pass:
        outputs = model(inputs)
        # 2. Compute loss:
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        # 3. Backward pass:
        loss.backward()
        # 4. Update weights:
        optimizer.step()

    # Compute average loss over all batches:
    losses = torch.tensor(losses)
    return torch.mean(losses).item()
