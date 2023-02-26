import numpy as np


def reorder_solution_array(arr: np.ndarray, swaps: list):
    """

    Parameters
    ----------
    arr: np.ndarray
        The array of solutions to be sorted

    swaps: list
        A list of tuples containing column swaps. These are necessary since each swap
        changes the position of a solution, so, if we want to recover the original order
        (x1, x2, ..., xn) we must pass explicitly this list of swaps.

    Returns
    -------
    arr: np.ndarray
        The array of solutions correctly sorted
    """
    if swaps:
        for i, j in swaps[::-1]:
            tmp = arr[j]
            arr[j] = arr[i]
            arr[i] = tmp
    return arr


def reorder_solution_matrix(A: np.ndarray, swaps: list):
    """

    Parameters
    ----------
    A: np.ndarray
        The matrix to be sorted

    swaps: list
        A list of tuples containing column swaps. These are necessary since each swap
        changes the position of a solution, so, if we want to recover the original order
        (x1, x2, ..., xn) we must pass explicitly this list of swaps.

    Returns
    -------
    A: np.ndarray
        The matrix correctly sorted
    """
    if swaps:
        for i, j in swaps[::-1]:
            A[[i, j]] = A[[j, i]]
    return A


def get_max_pivot_value_row(A: np.ndarray, col_index: int, n_rows: int):
    """
    Gets the index of the row that contains the maximum (absolute) value
    Parameters
    ----------
    A: np.ndarray
        The input matrix
    col_index: int
        The row index
    n_rows: int
        The number of rows

    Returns
    -------
    max_value_index: int
        The row index
    """
    max_value_index = col_index
    max_value = abs(A[col_index, col_index])
    for i in range(col_index + 1, n_rows):
        curr = abs(A[i, col_index])
        if curr > max_value:
            max_value = curr
            max_value_index = i
    return max_value_index


def get_max_pivot_value_row_col(
    A: np.ndarray, row_index: int, col_index: int, n_rows: int
):
    """
    Gets col and row index that contains the maximum (absolute) value

    Parameters
    ----------
    A: np.ndarray
        The input matrix
    row_index: int
        The row index
    col_index: int
        The col index
    n_rows: int
        The number of rows

    Returns
    -------
    max_value_row_index: int
        The row index
    min_value_row_index: int
        The col index
    """
    A = A[row_index:, col_index:n_rows]
    _row_index, _col_index = np.where(abs(A.max()) == abs(A))
    return _row_index[0] + row_index, _col_index[0] + col_index
