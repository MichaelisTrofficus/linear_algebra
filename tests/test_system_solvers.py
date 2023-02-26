import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from linear_algebra.system_solvers import solve_linear_system


@pytest.mark.parametrize(
    "algorithm",
    [
        "gauss",
        "gauss_partial_pivoting",
        "gauss_total_pivoting",
        "gauss_jordan",
        "gauss_seidel_method",
        "lu",
    ],
)
def test_solve_linear_system(algorithm):
    result = np.array([1.0, 2.0, 3.0])

    if algorithm == "gauss":
        X = np.array([[2.0, -1.0, 1.0], [-1.0, 1.0, 2.0], [1.0, 2.0, -1.0]])

        b = np.array([[3.0], [7.0], [2.0]])
    elif algorithm == "gauss_seidel_method":
        X = np.array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])

        b = np.array([[7.0], [-21.0], [15.0]])
        result = np.array([2.0, 4.0, 3.0])
    else:
        X = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 5.0], [3.0, -1.0, -2.0]])

        b = np.array([[2.0], [25.0], [-5.0]])
    solutions = solve_linear_system(X, b, algorithm=algorithm)
    assert_array_almost_equal(solutions, result)
