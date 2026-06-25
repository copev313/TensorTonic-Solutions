def frequency_encoding(values):
    """Replace each value with its frequency proportion."""
    counts = {}
    total = len(values)
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    return [counts[v] / total for v in values]