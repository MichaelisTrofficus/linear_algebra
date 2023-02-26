import numpy as np
from linear_algebra.utils import reorder_solution_array
from linear_algebra.reductions import (
    gauss,
    gauss_partial_pivoting,
    gauss_total_pivoting,
    gauss_jordan,
    gauss_seidel_method,
)
from linear_algebra.factorization import lu


def back_substitution(A: np.ndarray):
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
    solutions = np.zeros(m)
    solutions[m - 1] = A[m - 1, n - 1] / A[m - 1, n - 2]
    for i in range(m - 2, -1, -1):
        solutions[i] = (
            A[i, n - 1] - np.dot(A[i, i + 1 : n - 1], solutions[i + 1 :])
        ) / A[i, i]
    return solutions


def forward_substitution(A: np.ndarray):
    """
    Forward Substitution method for calculating the solutions of a matrix in row echelon form
    Parameters
    ----------
    A: numpy.ndarray
        A matrix in row echelon form

    Returns
    -------
    numpy.ndarray
        An array containing the solutions to the system of linear equations
    """
    m, n = A.shape
    solutions = np.zeros(m)
    solutions[0] = A[0, n - 1] / A[0, 0]
    for i in range(1, m):
        solutions[i] = (A[i, n - 1] - np.dot(A[i, : i + 1], solutions[: i + 1])) / A[
            i, i
        ]
    return solutions


def solve_linear_system(
    A: np.ndarray, b: np.ndarray, algorithm="gauss_jordan", **kwargs
):
    """
    Solves a system of linear equations

    Parameters
    ----------
    A: np.ndarray
        The coefficient matrix
    b: np.ndarray

    algorithm: str
        The algorithm to use for solving the system

    Returns
    -------
    solution:
        An array representing the solutions to the system
    """
    A_augmented = np.concatenate([A, b], axis=1)
    m, n = A_augmented.shape

    if algorithm == "gauss":
        return back_substitution(gauss(A_augmented))
    elif algorithm == "gauss_partial_pivoting":
        A_ref, _ = gauss_partial_pivoting(A_augmented)
        return back_substitution(A_ref)
    elif algorithm == "gauss_total_pivoting":
        A_ref, swaps = gauss_total_pivoting(A_augmented)
        return reorder_solution_array(back_substitution(A_ref), swaps)
    elif algorithm == "gauss_jordan":
        A_ref, swaps = gauss_jordan(A_augmented)
        return reorder_solution_array(A_ref[:, n - 1], swaps)
    elif algorithm == "gauss_seidel_method":
        return gauss_seidel_method(A, b, **kwargs)
    elif algorithm == "lu":
        L, U, P = lu(A)

        # 1. Ly = Pb
        P_b = np.reshape(np.dot(P, b), (-1, 1))
        L_augmented = np.concatenate([L, P_b], axis=1)
        y = np.reshape(forward_substitution(L_augmented), (-1, 1))

        # 2. Ux = y
        U_augmented = np.concatenate([U, y], axis=1)
        return back_substitution(U_augmented)

    else:
        raise ValueError("Algorithm not implemented")


X_1 = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 5.0], [3.0, -1.0, -2.0]])

b_1 = np.array([[2.0], [25.0], [-5.0]])

solutions = solve_linear_system(X_1, b_1, algorithm="gauss_total_pivoting")
print(solutions)
