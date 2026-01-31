from math import log1p

def log_transform(values):
    """Apply the log1p transformation to each value.

    Parameters:
    -----------
    values: list[int | float]
        A list of numeric values.

    Returns:
    --------
    list[float]
        Calculation of log1p of each input value.
    """
    return [log1p(v) for v in values]