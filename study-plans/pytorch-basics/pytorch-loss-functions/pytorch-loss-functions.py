import torch


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    method: str,
    delta: float = 1.0,
):
    """Computes the loss between the predicted and target values.

    Parameters:
    -----------
    pred: torch.Tensor 
        The predicted values.
    target: torch.Tensor
        The target values.
    method: str
        The loss method to use. Supported values are "mse", "huber", and "cross_entropy".
    delta: float, optional (default=1.0)
        The delta value for the Huber loss. Ignored for other loss methods.

    Returns:
    --------
    float
        The mean loss value.
    """
    pred = torch.Tensor(pred)
    target = torch.Tensor(target)
    
    match method:
        case "mse":
            return torch.mean((pred - target) ** 2).item()

        case "huber":
            error = pred - target
            abs_error = torch.abs(error)
            loss = torch.where(
                abs_error <= delta,
                0.5 * error ** 2,
                delta * (abs_error - 0.5 * delta)
            )
            return torch.mean(loss).item()

        case "cross_entropy":
            # Use log-sum-exp trick for numerical stability:
            max_pred = torch.max(pred, dim=1, keepdim=True).values
            log_sum_exp = torch.log(
                torch.sum(
                    torch.exp(pred - max_pred), 
                    dim=1, 
                    keepdim=True,
                )
            ) + max_pred
            # Use log_softmax then index with target:
            log_softmax = pred - log_sum_exp
            return -torch.mean(log_softmax[torch.arange(pred.size(0)), target.long()]).item()

        case _:
            raise ValueError(f"Unsupported loss method: {method}")
