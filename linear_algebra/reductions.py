import numpy as np


def get_max_pivot_value_row(A, i, m):
    max_value_index = i
    max_value = abs(A[i, i])
    for j in range(i + 1, m):
        curr = abs(A[j, i])
        if curr > max_value:
            max_value = curr
            max_value_index = j
    return max_value_index


def get_max_pivot_value_coefficients(A, i, j, m):
    A = A[i:, j:m]
    row_index, col_index = np.where(abs(A.max()) == abs(A))
    return row_index[0] + i, col_index[0] + j


def reorder_solution_array(arr, swaps):
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


def gauss(A):
    """
    Implements a simplified version of Gauss elimination method

    Parameters
    ----------
    A: numpy.ndarray
        The augmented matrix for a system of linear equations

    Returns
    -------
    """
    m, n = A.shape
    i = j = 0

    while i < m and j < n:
        for k in range(j + 1, m):
            A[k:] = A[k:] - (A[k, i] / A[i, i]) * A[i, :]
        i += 1
        j += 1
    return A


def gauss_partial_pivoting(A):
    """
    Implements a simplified version of Gauss elimination method with partial pivoting
    Parameters
    ----------
    A: numpy.ndarray

    Returns
    -------
    """
    m, n = A.shape
    i = j = 0
    swaps = []

    while i < n and j < m:
        max_index = get_max_pivot_value_row(A, i, m)

        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            swaps.append((i, max_index))

        for k in range(j + 1, m):
            A[k, :] = A[k, :] - (A[k, i] / A[i, i]) * A[i, :]

        i += 1
        j += 1

    return A, swaps


def gauss_total_pivoting(A):
    """

    Parameters
    ----------
    A

    Returns
    -------

    """
    m, n = A.shape
    i = j = 0
    swaps = []

    while i < n and j < m:
        max_row_index, max_col_index = get_max_pivot_value_coefficients(A, i, j, m)
        A[[j, max_row_index]] = A[[max_row_index, j]]
        A[:, [i, max_col_index]] = A[:, [max_col_index, i]]

        if max_col_index != i:
            swaps.append((i, max_col_index))

        for k in range(j + 1, m):
            A[k, :] = A[k, :] - (A[k, i] / A[i, i]) * A[i, :]

        i += 1
        j += 1

    return A, swaps


def gauss_jordan(A):
    """

    Parameters
    ----------
    A
    get_solutions

    Returns
    -------

    """
    A_ref, swaps = gauss_total_pivoting(A)

    m, n = A_ref.shape
    i = m - 1

    while i >= 0:
        A_ref[i, :] = A_ref[i, :] / A_ref[i, i]
        for k in range(i - 1, -1, -1):
            A_ref[k, :] = A_ref[k, :] - (A_ref[k, i] / A_ref[i, i]) * A_ref[i, :]
        i -= 1
    return A_ref, swaps


def gauss_seidel_method(A, b, max_iter=10):
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
