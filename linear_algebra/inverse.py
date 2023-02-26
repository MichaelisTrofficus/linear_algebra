import numpy as np
from reductions import gauss_jordan


def reorder_solution_array(arr, swaps):
    if swaps:
        for i, j in swaps[::-1]:
            tmp = arr[j]
            arr[j] = arr[i]
            arr[i] = tmp
    return arr


def reorder_solution_matrix(A, swaps):
    if swaps:
        for i, j in swaps[::-1]:
            A[[i, j]] = A[[j, i]]
    return A


def gauss_jordan_inverse(A):
    """

    Parameters
    ----------
    A: numpy.ndarray
        The matrix to calculate the inverse from. We are assuming it is a squared matrix or it
        will raise an error.

    Returns
    -------
    """
    m, n = A.shape

    if m != n:
        raise ValueError("Input matrix must be square!")

    A = np.concatenate([A, np.identity(m)], axis=1)
    A, swaps = gauss_jordan(A)
    A_inv = reorder_solution_matrix(A[:, n:], swaps)
    return A_inv


X = np.array([
    [1., 2., -1.],
    [2., 4., 5.],
    [3., -1., -2.]
])

X_1 = gauss_jordan_inverse(X)
print(np.dot(X, X_1))

