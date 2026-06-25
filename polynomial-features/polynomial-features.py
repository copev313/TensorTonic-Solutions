def polynomial_features(values, degree):
    """Generate polynomial features for each value up to the given degree."""
    return [
        [v ** d for d in range(degree + 1)] 
        for v in values
    ]
