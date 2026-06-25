def interaction_features(X):
    """Generate pairwise interaction features and append them to the original features."""
    feats = []
    for sample in X:
        row = list(sample)
        interactions = []
        d = len(row)
        for i in range(d):
            for j in range(i + 1, d):
                interactions.append(row[i] * row[j])
        feats.append(row + interactions)
    return feats