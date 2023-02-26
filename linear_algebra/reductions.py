import numpy as np


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


def reorder_solution_array(arr: np.ndarray, swaps: list):
    """

    Parameters
    ----------
    arr
    swaps

    Returns
    -------

    """
    if swaps:
        for i, j in swaps[::-1]:
            tmp = arr[j]
            arr[j] = arr[i]
            arr[i] = tmp
    return arr


def back_substitution(A, swaps=None, rref=False):
    """
    Back Substitution method for calculating the solutions of a matrix in row echelon form
    Parameters
    ----------
    A: numpy.ndarray
        A matrix in row echelon form
    swaps: list
        A list of tuples containing column swaps. These are necessary since each swap
        changes the position of a solution, so, if we want to recover the original order
        (x1, x2, ..., xn) we must pass explicitly this list of swaps.
    rref: bool
        If the input matrix is in reduced row echelon form simply take the last column.

    Returns
    -------
    numpy.ndarray
        An array containing the solutions to the system of linear equations
    """
    m, n = A.shape
    if rref:
        solutions = A[:, n - 1]
    else:
        solutions = np.zeros(m)
        solutions[m - 1] = A[m - 1, n - 1] / A[m - 1, n - 2]
        for i in range(m - 2, -1, -1):
            solutions[i] = (
                A[i, n - 1] - np.dot(A[i, i + 1 : n - 1], solutions[i + 1 :])
            ) / A[i, i]
    return reorder_solution_array(solutions, swaps)


def forward_substitution(A, swaps=None, rref=False):
    """
    Back Substitution method for calculating the solutions of a matrix in row echelon form
    Parameters
    ----------
    A: numpy.ndarray
        A matrix in row echelon form
    swaps: list
        A list of tuples containing column swaps. These are necessary since each swap
        changes the position of a solution, so, if we want to recover the original order
        (x1, x2, ..., xn) we must pass explicitly this list of swaps.
    rref: bool
        If the input matrix is in reduced row echelon form simply take the last column.

    Returns
    -------
    numpy.ndarray
        An array containing the solutions to the system of linear equations
    """
    m, n = A.shape
    if rref:
        solutions = A[:, n - 1]
    else:
        solutions = np.zeros(m)
        solutions[0] = A[0, n - 1] / A[0, 0]
        for i in range(1, m):
            solutions[i] = (
                A[i, n - 1] - np.dot(A[i, : i + 1], solutions[: i + 1])
            ) / A[i, i]
    return reorder_solution_array(solutions, swaps)


def gauss(A: np.ndarray):
    """
    Implements a simplified version of Gauss elimination method

    Parameters
    ----------
    A: numpy.ndarray
        The augmented matrix for a system of linear equations

    Returns
    -------
    A: numpy.ndarray
        The augmented matrix in row echelon form
    """
    m, n = A.shape

    for i in range(m):
        for j in range(i + 1, m):
            A[j:] = A[j:] - (A[j, i] / A[i, i]) * A[i, :]
    return A


def gauss_partial_pivoting(A: np.ndarray):
    """
    Implements a simplified version of Gauss elimination method with partial pivoting
    Parameters
    ----------
    A: numpy.ndarray
        The augmented matrix for a system of linear equations

    Returns
    -------
    A: np.ndarray
        The input matrix in row echelon form

    swaps: list
        A list of tuples, where each tuple represents a swap operation between rows
    """
    m, n = A.shape
    swaps = []

    for i in range(m):
        max_index = get_max_pivot_value_row(A, i, m)

        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            swaps.append((i, max_index))

        for j in range(i + 1, m):
            A[j, :] = A[j, :] - (A[j, i] / A[i, i]) * A[i, :]

    return A, swaps


def gauss_total_pivoting(A):
    """
    Gauss method with total pivoting

    Parameters
    ----------
    A: np.ndarray
        A matrix

    Returns
    -------
    A: np.ndarray
        The input matrix in row echelon form

    swaps: list
        A list of tuples, where each tuple represents a swap operation between rows
    """
    m, n = A.shape
    i = j = 0
    swaps = []

    while i < n and j < m:
        max_row_index, max_col_index = get_max_pivot_value_row_col(A, i, j, m)
        A[[j, max_row_index]] = A[[max_row_index, j]]
        A[:, [i, max_col_index]] = A[:, [max_col_index, i]]

        if max_col_index != i:
            swaps.append((i, max_col_index))

        for k in range(j + 1, m):
            A[k, :] = A[k, :] - (A[k, i] / A[i, i]) * A[i, :]

        i += 1
        j += 1

    return A, swaps


def gauss_jordan(A: np.ndarray):
    """
    Gauss Jordan method

    Parameters
    ----------
    A: np.ndarray
        A matrix

    Returns
    -------
    A_rref: np.ndarray
        The input matrix in reduced row echelon form

    swaps: list
        A list of tuples, where each tuple represents a swap operation between rows
    """
    A_rref, swaps = gauss_total_pivoting(A)

    m, n = A_rref.shape
    i = m - 1

    while i >= 0:
        A_rref[i, :] = A_rref[i, :] / A_rref[i, i]
        for k in range(i - 1, -1, -1):
            A_rref[k, :] = A_rref[k, :] - (A_rref[k, i] / A_rref[i, i]) * A_rref[i, :]
        i -= 1
    return A_rref, swaps


def gauss_seidel_method(A, b, max_iter=10):
    """

    Parameters
    ----------
    A
    b
    max_iter

    Returns
    -------

    """
    m, n = A.shape
    solutions = np.zeros(m)
    it = 0

    while it < max_iter:
        for i in range(m):
            if i == 0:
                left_sum = 0
                right_sum = np.dot(A[i, i + 1 :], solutions[i + 1 :])
            elif i == m - 1:
                right_sum = 0
                left_sum = np.dot(A[i, :i], solutions[:i])
            else:
                right_sum = np.dot(A[i, i + 1 :], solutions[i + 1 :])
                left_sum = np.dot(A[i, :i], solutions[:i])

            solutions[i] = (b[i] - right_sum - left_sum) / A[i, i]

        print(solutions)
        it += 1

    return solutions


X1 = np.array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])

b1 = np.array([7, -21, 15])

X1_ref = gauss_seidel_method(X1, b1)
