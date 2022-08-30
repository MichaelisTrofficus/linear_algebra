import numpy as np


def list_row_pivots(m: np.ndarray) -> list:
    """
    Simply lists the row pivots a matrix has
    Args:
        m: A matrix

    Returns:
        A list of pivots indices
    """
    r, c = m.shape
    null_rows = list(np.where(~m.any(axis=1))[0])

    pivots = []
    for r_index in range(r - len(null_rows)):
        non_zero_index = (m[r_index] != 0).argmax()
        pivots.append([r_index, non_zero_index])
    return pivots


def has_row_echelon_form(m: np.ndarray) -> bool:
    """
    Checks if the given matrix `m` has row echelon form
    Args:
        m: A numpy array representing a matrix

    Returns:
        A boolean indicating if the given matrix is in row echelon form
    """
    if len(m.shape) == 1:
        # It is a row matrix
        return True

    r, c = m.shape
    null_rows = list(np.where(~m.any(axis=1))[0])
    last_rows = [x for x in range(r - len(null_rows), r, 1)]

    if last_rows != null_rows:
        return False

    prev_non_zero_index = (m[0] != 0).argmax()
    for r_index in range(1, r - len(null_rows)):
        non_zero_index = (m[r_index] != 0).argmax()
        if non_zero_index <= prev_non_zero_index:
            return False
        prev_non_zero_index = non_zero_index

    return True


def has_reduced_row_echelon_form(m: np.ndarray) -> bool:
    """
    Checks if the given matrix `m` has reduced row echelon form
    Args:
        m: A numpy array representing a matrix

    Returns:
        A boolean indicating if the given matrix is in reduced row echelon form
    """
    if not has_row_echelon_form(m):
        return False

    r, c = m.shape
    null_rows = list(np.where(~m.any(axis=1))[0])

    for r_index in range(r - len(null_rows)):
        c_index = (m[r_index] != 0).argmax()
        if m[r_index, c_index] != 1 or sum(m[:, c_index]) != 1:
            return False
    return True


def gauss(m: np.ndarray) -> np.ndarray:
    """
    Implementation of Gauss algorithm using recursion
    Args:
        m: A matrix

    Returns:
        A matrix in row echelon form
    """
    last_step = False

    if has_row_echelon_form(m):
        return m
    else:
        first_non_zero_column = (m != 0).any(axis=0).argmax()
        first_non_zero_row = (m[:, first_non_zero_column] != 0).argmax()
        m[[0, first_non_zero_row]] = m[[first_non_zero_row, 0]]

        pivot = m[0, first_non_zero_column]

        for row_index in range(1, m.shape[0]):
            element = m[row_index, first_non_zero_column]
            if element != 0:
                m[row_index, :] = m[row_index, :] - (element / pivot) * m[0, :]

        sub_matrix = m[1:, first_non_zero_column + 1:]

        if not sub_matrix.size:
            # Last step
            sub_matrix = m
            last_step = True

        row_echelon_sub_matrix = gauss(sub_matrix)

        if not last_step:
            m[1:, first_non_zero_column + 1:] = row_echelon_sub_matrix
        else:
            m[:] = row_echelon_sub_matrix

        return m


def gauss_jordan(m: np.ndarray) -> np.ndarray:
    """
    An implementation of Gauss-Jordan algorithm
    Args:
        m: A matrix

    Returns:
        A matrix in reduced row echelon form
    """
    if not has_row_echelon_form(m):
        m = gauss(m)

    pivots_indices = list_row_pivots(m)[::-1]
    for r_pivot, c_pivot in pivots_indices:

        pivot = m[r_pivot, c_pivot]
        m[r_pivot, :] = 1 / pivot * m[r_pivot, :]

        for row_index in range(r_pivot):
            element = m[row_index, c_pivot]
            if element != 0:
                m[row_index, :] = m[row_index, :] - element * m[r_pivot, :]
    return m

