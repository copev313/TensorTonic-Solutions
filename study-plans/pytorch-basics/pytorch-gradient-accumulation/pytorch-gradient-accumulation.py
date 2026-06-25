import torch


def gradient_accumulation(
    w_init: list[float],
    micro_batches: list[tuple], 
    lr: float,
    accum_steps: int,
) -> tuple:
    """Accumulate gradients across micro-batches.
    
    Parameters:
    -----------
    w_init: list[float]
        Initial weights
    micro_batches: list[tuple]
        List of micro-batches, where each micro-batch is a tuple of (inputs, targets).
    lr: float
        Learning rate for weight updates.
    accum_steps: int
        Number of micro-batches to accumulate gradients over before updating weights.
    
    Returns: 
    --------
    tuple
        A tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
    accum_grad = torch.zeros_like(w)
    updated_weights = []
    last_avg_grad = None

    for i, (inputs, targets) in enumerate(micro_batches):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        outputs = inputs @ w
        loss = torch.mean((outputs - targets) ** 2)
        loss.backward()

        accum_grad += w.grad.detach().clone()
        w.grad.zero_()

        if (i + 1) % accum_steps == 0:
            last_avg_grad = accum_grad / accum_steps
            with torch.no_grad():
                w -= lr * last_avg_grad
            
            updated_weights.append(w.detach().clone().tolist())
            accum_grad = torch.zeros_like(w)

    return (
        updated_weights[-1] if updated_weights else [],
        last_avg_grad.tolist() if last_avg_grad is not None else None
    )
