import numpy as np
from linear_algebra.reductions import gauss_jordan
from linear_algebra.utils import reorder_solution_matrix


def gauss_jordan_inverse(A: np.ndarray):
    """

    Parameters
    ----------
    A: numpy.ndarray
        The matrix to calculate the inverse from. We are assuming it is a squared matrix or it
        will raise an error.

    Returns
    -------
    A_inv: np.ndarray
        The inverse of input matrix A
    """
    m, n = A.shape

    if m != n:
        raise ValueError("Input matrix must be square!")

    A = np.concatenate([A, np.identity(m)], axis=1)
    A, swaps = gauss_jordan(A)
    A_inv = reorder_solution_matrix(A[:, n:], swaps)
    return A_inv
