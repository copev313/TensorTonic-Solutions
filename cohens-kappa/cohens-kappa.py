import numpy as np


def cohens_kappa(rater1, rater2):
    """Compute Cohen's Kappa coefficient."""
    r1 = np.asarray(rater1).ravel()
    r2 = np.asarray(rater2).ravel()

    if r1.size == 0 or r1.shape != r2.shape:
        raise ValueError("Rater arrays must have the same non-zero length.")

    observed_agreement = np.mean(r1 == r2)

    categories = np.unique(np.concatenate((r1, r2)))
    expected_agreement = 0.0
    for category in categories:
        p1 = np.mean(r1 == category)
        p2 = np.mean(r2 == category)
        expected_agreement += p1 * p2

    if np.isclose(expected_agreement, 1.0):
        return 1.0 if np.isclose(observed_agreement, 1.0) else 0.0

    return float((observed_agreement - expected_agreement) / (1 - expected_agreement))