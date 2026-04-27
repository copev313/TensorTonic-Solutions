# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.7",
#     "numpy>=2.4.0",
#     "pytest>=9.0.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells:
    import marimo as mo
    import numpy as np
    import pytest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Sigmoid Function
    """)
    return


@app.function
# INPUT: scalars, Python lists, or NumPy arrays
# OUTPUT: NumPy array of floats
# Must be vectorized


def sigmoid(x):
    """Vectorized sigmoid function.

    Parameters
    ----------
    x: int | float | list | np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Sigmoid values.
    """
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


@app.class_definition
class TestSigmoid:
    def test_example_01(self):
        x = [0, 2, -2]
        expected = np.array([0.5, 0.88079708, 0.11920292])
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-6)

    def test_example_02(self):
        x = 0
        expected = np.array([0.5])
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-6)

    def test_example_03(self):
        x = [[-1, 0], [1, 2]]
        expected = np.array([[0.26894142, 0.5], [0.73105858, 0.88079708]])
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-6)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Dot Product
    """)
    return


@app.function
# INPUT: lists or NumPy arrays
# OUTPUT: float
# Must be vectorized


def dot_product(x, y) -> float:
    """Compute the dot product of two 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sum(x * y)


@app.class_definition
class TestDotProduct:
    x1, y1 = [1, 2, 3], [4, 5, 6]
    x2, y2 = np.array([1, 0]), np.array([0, 1])
    x3, y3 = np.array([-1, 2]), np.array([3, -1])

    def test_example_01(self):
        assert dot_product(self.x1, self.y1) == 32.0

    def test_example_02(self):
        assert dot_product(self.x2, self.y2) == 0.0

    def test_example_03(self):
        assert dot_product(self.x3, self.y3) == -5.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Euclidean Distance
    """)
    return


@app.function
# INPUT: lists or NumPy arrays
# OUTPUT: float
# Must be vectorized


def euclidean_distance(x, y):
    """Compute the Euclidean (L2) distance between vectors x and y.
    Returns result as a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(np.sum((x - y) ** 2))


@app.class_definition
class TestEuclideanDistance:
    x1, y1 = np.array([3, 4]), np.array([0, 0])
    x2, y2 = np.array([1, 2, 3]), np.array([4, 5, 6])
    x3, y3 = np.array([0, 0, 0]), np.array([0, 0, 0])

    def test_example_01(self):
        assert np.isclose(euclidean_distance(self.x1, self.y1), 5.0)

    def test_example_02(self):
        assert np.isclose(euclidean_distance(self.x2, self.y2), 5.19615)

    def test_example_03(self):
        assert euclidean_distance(self.x3, self.y3) == 0.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Mean Squared Error
    """)
    return


@app.function
# Convert inputs to NumPy arrays
# OUTPUT: a single float, return None if shapes do not match
# NumPy only


def mean_squared_error(y_pred, y_true):
    """Compute Mean Squared Error (MSE) between predicted and true values.

    Parameters:
    -----------
    y_pred: array-like
        Predicted values (list, scalar, or np.ndarray).
    y_true: array-like
        True values (list, scalar, or np.ndarray).

    Returns:
    --------
    float or None
        MSE calculation. Returns None if input shapes do not match.
    """
    y_hat = np.array(y_pred)
    y = np.array(y_true)
    # [CASE] Shape mismatch:
    if y_hat.shape != y.shape:
        return None

    return np.mean((y_hat - y) ** 2)


@app.class_definition
class TestMeanSquaredError:
    def test_example_01(self):
        y_pred = [2, 3]
        y_true = [1, 1]
        assert mean_squared_error(y_pred, y_true) == 2.5

    def test_example_02(self):
        y_pred = [0, 0, 0]
        y_true = np.array([0, 0, 0])
        assert mean_squared_error(y_pred, y_true) == 0.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Cosine Similarity
    """)
    return


@app.function
# INPUT: 1d numpy arrays of equal length
# OUTPUT: scalar float
# - Fully vectorized (no loops)
# - Handle zero vectors gracefully (return 0 if either norm is 0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between Numpy two arrays."""
    # Magnitudes of a and b:
    mag_prod = np.linalg.norm(a) * np.linalg.norm(b)
    return (np.dot(a, b) / mag_prod) if mag_prod != 0.0 else 0


@app.class_definition
class TestCosineSimilarity:
    a, b = np.array([1, 2, 3]), np.array([2, 4, 6])
    c, d = np.array([1, 2]), np.array([0, 0, 0])
    e, f = np.array([1, 0]), np.array([0, 1])

    def test_zero_magnitude(self):
        assert cosine_similarity(self.b, self.d) == 0

    def test_example_01(self):
        assert np.isclose(cosine_similarity(self.a, self.b), 1.0)

    def test_example_02(self):
        assert np.isclose(cosine_similarity(self.e, self.f), 0.0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Covariance Matrix
    """)
    return


@app.function
# OUTPUT: ndarray of shape (D, D) with covariance values or None for invalid input (N < 2 or not 2D)
# Must be vectorized
# Cannot use `np.cov` function
# NOTE: Use sample covariance (divide by N-1, not N)


def covariance_matrix(X):
    """Compute covariance matrix from dataset X.

    Parameters
    ----------
    X: list[list[float]] | np.ndarray
        Dataset with shape (N, D)

    Returns
    -------
    np.ndarray | None
        Covariance matrix with shape (D, D). None for invalid input (N < 2 or not 2D)
    """
    X = np.asarray(X, dtype=float)
    # [CASE] Invalid input dims:
    if X.ndim != 2:
        return None

    N = X.shape[0]
    if N < 2:
        return None

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov_mat = (X_centered.T @ X_centered) / (N - 1)
    return cov_mat


@app.class_definition
class TestCovarianceMatrix:
    def test_example_01(self):
        X = [[1, 2], [2, 3], [3, 4]]
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(covariance_matrix(X), expected, rtol=1e-6)

    def test_example_02(self):
        X = [[1, 0], [0, 1]]
        expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_allclose(covariance_matrix(X), expected, rtol=1e-6)

    def test_example_03(self):
        X = [[1, 2, 3]]
        assert covariance_matrix(X) is None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Cross Entropy Loss
    """)
    return


@app.function
# Must be vectorized
# INPUT: probabilities (not logits).
# NOTES:
# - All probabilities are guaranteed to be > 0
# - y_true and y_pred must have matching first dimension N
# - y_true contains valid class indices for the second dimension of y_pred


def cross_entropy_loss(y_true, y_pred):
    """Compute average cross-entropy loss for multi-class classification.

    Parameters
    ----------
    y_true: np.ndarray
        True class labels. Shape: (N,)
    y_pred: np.ndarray
        Predicted probabilities for each class. Shape: (N, C)

    Returns
    -------
    float
        Average CE loss over all samples.
    """
    y_hat = np.asarray(y_pred)
    y = np.asarray(y_true)
    length = y.shape[0]
    # Select predicted probs for true classes:
    select = y_hat[np.arange(length), y]
    # Calc. neg avg log likelihood:
    return -np.mean(np.log(select))


@app.class_definition
class TestCrossEntropyLoss:
    y_true1 = np.array([0, 1])
    y_pred1 = np.array([[0.9, 0.1], [0.3, 0.7]])

    y_true2 = np.array([2])
    y_pred2 = np.array([[0.1, 0.1, 0.8]])

    y_true3 = np.array([1, 0, 1])
    y_pred3 = np.array([[0.2, 0.8], [0.6, 0.4], [0.49, 0.51]])

    def test_example_01(self):
        assert np.isclose(
            cross_entropy_loss(self.y_true1, self.y_pred1), 0.231018, rtol=1e-5
        )

    def test_example_02(self):
        assert np.isclose(
            cross_entropy_loss(self.y_true2, self.y_pred2), 0.223144, rtol=1e-5
        )

    def test_example_03(self):
        assert np.isclose(
            cross_entropy_loss(self.y_true3, self.y_pred3), 0.469105, rtol=1e-5
        )


if __name__ == "__main__":
    app.run()
