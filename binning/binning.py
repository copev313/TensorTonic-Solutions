import math

def binning(values, num_bins):
    """Assign each value to an equal-width bin."""
    # 1. Calc bin width:
    minimum = min(values)
    maximum = max(values)
    # Edge case:
    if minimum == maximum:
        return [0 for v in values]
    width = (maximum - minimum) / num_bins
    # 2. Assign values to bins:
    return [
        min(math.floor((v - minimum) / width), num_bins - 1) 
        for v in values
    ]
