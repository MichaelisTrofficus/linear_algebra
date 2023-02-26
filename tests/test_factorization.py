import numpy as np
from numpy.testing import assert_array_almost_equal

from linear_algebra.factorization import lu


def test_lu():
    X = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 5.0], [3.0, -1.0, -2.0]])
    L, U, P = lu(X)

    assert_array_almost_equal(
        L, np.array([[1.0, 0.0, 0.0], [0.66666667, 1.0, 0.0], [0.33333333, 0.5, 1.0]])
    )
    assert_array_almost_equal(
        U,
        np.array([[3.0, -1.0, -2.0], [0.0, 4.66666667, 6.33333333], [0.0, 0.0, -3.5]]),
    )
    assert_array_almost_equal(
        P, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )
