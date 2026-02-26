import numpy as np

LAMBDA = 1.0507009873554804934193349852946
ALPHA = 1.6732632423543772848170429916717


def selu(
    x,
    lam=LAMBDA,
    alpha=ALPHA,
):
    """Applies the SELU activation function element-wise to the input
    list of floats.

    Parameters
    ----------
    x: list of floats
        The input values to which the SELU activation function will be applied.
    lam: float, optional (default=LAMBDA)
        The lambda parameter for the SELU function.
    alpha: float, optional (default=ALPHA)
        The alpha parameter for the SELU function.

    Returns
    -------
    list[float]
        A list of floats rounded to 4 decimal places.
    """
    x_arr = np.array(x, dtype=float)
    activated = np.where(
        x_arr > 0,
        lam * x_arr,
        lam * alpha * (np.exp(x_arr) - 1),
    )
    return np.round(activated, 4).tolist()
