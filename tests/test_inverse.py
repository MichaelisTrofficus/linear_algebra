import numpy as np
from numpy.testing import assert_array_almost_equal

from linear_algebra.inverse import gauss_jordan_inverse


def test_gauss_jordan_inverse():
    X = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 5.0], [3.0, -1.0, -2.0]])
    X_inverse = gauss_jordan_inverse(X)
    assert_array_almost_equal(np.dot(X, X_inverse), np.identity(X.shape[0]))
