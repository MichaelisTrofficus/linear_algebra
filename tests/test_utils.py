import numpy as np
from numpy.testing import assert_array_equal
from linear_algebra.utils import (
    get_max_pivot_value_row,
    get_max_pivot_value_row_col,
    reorder_solution_array,
)


def test_get_max_pivot_value_row():
    X = np.array([[4.0, -1.0, 1.0], [12.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
    assert get_max_pivot_value_row(X, col_index=0, n_rows=3) == 1


def test_get_max_pivot_value_row_col():
    X = np.array([[4.0, -1.0, 1.0], [12.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
    assert get_max_pivot_value_row_col(X, row_index=0, col_index=0, n_rows=3) == (1, 0)


def test_reorder_solution_array():
    arr = np.array([3.0, 1.0, 2.0])
    swaps = [(0, 2), (1, 2)]
    assert_array_equal(reorder_solution_array(arr, swaps), np.array([1.0, 2.0, 3.0]))
