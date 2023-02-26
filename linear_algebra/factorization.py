from copy import deepcopy
import numpy as np

from reductions import get_max_pivot_value_row, back_substitution, forward_substitution


def lu(A, b):
    """
    LU Factorization of the provided matrix.

    Parameters
    ----------
    A

    b

    Returns
    -------

    """
    m, n = A.shape
    i = j = 0

    # Permutation matrix
    P = np.identity(n)
    L = np.identity(n)
    U = deepcopy(A)

    while i < n and j < m:
        max_index = get_max_pivot_value_row(U, i, m)

        if max_index != i:
            U[[i, max_index]] = U[[max_index, i]]
            P[[i, max_index]] = P[[max_index, i]]

        for k in range(j + 1, m):
            mult = (U[k, i] / U[i, i])
            L[k, i] = mult
            U[k, :] = U[k, :] - mult * U[i, :]

        i += 1
        j += 1

    # Now that L, U and P are generated, we simply solve the equations

    # 1. Ly = Pb
    P_b = np.reshape(np.dot(P, b.T), (-1, 1))
    L_augmented = np.concatenate([L, P_b], axis=1)
    y = np.reshape(forward_substitution(L_augmented), (-1, 1))

    # 2. Ux = y
    U_augmented = np.concatenate([U, y], axis=1)
    x = back_substitution(U_augmented)

    return x


X_1 = np.array([
    [1., 2., -1.],
    [2., 4., 5.],
    [3., -1., -2.]
])

b_1 = np.array([2., 25., -5.])

solutions = lu(X_1, b_1)
print(solutions)


