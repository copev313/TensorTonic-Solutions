import math

def cyclic_encoding(values, period):
    """Encode cyclic features as sin/cos pairs."""
    # Compute angles in radians:
    angles = [(2 * math.pi * v / period) for v in values]
    return [[math.sin(a), math.cos(a)] for a in angles]
