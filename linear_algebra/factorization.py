from copy import deepcopy
import numpy as np

from linear_algebra.reductions import get_max_pivot_value_row


def lu(A: np.ndarray):
    """
    LU factorization of the provided matrix

    Parameters
    ----------
    A: np.ndarray
        The input matrix

    Returns
    -------
    L: np.ndarray
        Lower triangular matrix
    U: np.ndarray
        Upper triangular matrix
    P: np.ndarray
        Permutation matrix
    """
    m, n = A.shape

    # Permutation matrix
    P = np.identity(n)
    L = np.identity(n)
    U = deepcopy(A)

    for i in range(m):
        max_index = get_max_pivot_value_row(U, i, m)

        if max_index != i:
            U[[i, max_index]] = U[[max_index, i]]
            P[[i, max_index]] = P[[max_index, i]]

        for j in range(i + 1, m):
            mult = U[j, i] / U[i, i]
            L[j, i] = mult
            U[j, :] = U[j, :] - mult * U[i, :]

    return L, U, P
