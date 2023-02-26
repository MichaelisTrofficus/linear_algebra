import numpy as np
from linear_algebra.utils import get_max_pivot_value_row, get_max_pivot_value_row_col


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


def gauss_seidel_method(A, b, max_iter=10, verbose=0):
    """
    Implementation of the Gauss Seidel method, a class of iterative methods.

    Parameters
    ----------
    A: np.ndarray
        The matrix of coefficients
    b: np.ndarray

    max_iter: int
        The maximum number of iteration

    verbose: int
        Parameter to set the verbosity of the iterative process

    Returns
    -------
    solutions: np.ndarray
        An array containing the solutions
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

            solutions[i] = (b[i][0] - right_sum - left_sum) / A[i, i]

        if verbose > 0:
            print(solutions)
        it += 1

    return solutions
