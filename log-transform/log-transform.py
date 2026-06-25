import math

def log_transform(values):
    """Apply the log1p transformation to each value."""
    return [math.log1p(value) for value in values]
