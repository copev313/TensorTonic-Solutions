import torch
import torch.nn as nn


def manual_train_step(model, X, y, criterion, lr):
    """Finds the loss from a single training step w/o using an optimizer.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.
    X : torch.Tensor
        The input data.
    y : torch.Tensor
        The target labels.
    criterion : torch.nn.Module
        The loss function to use.
    lr : float
        The learning rate for the manual parameter update.

    Returns: 
    --------
    float
        The loss value
    """
    # Forward pass:
    y_pred = model(X)
    # Compute loss:
    loss = criterion(y_pred, y)
    # Backward pass (compute gradients):
    loss.backward()
    # Update parameters manually:
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # Zero gradients after updating:
    model.zero_grad()
    return loss.item()
