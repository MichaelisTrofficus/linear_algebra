import numpy as np
from numpy.testing import assert_array_equal


def back_substitution(X, b):
    n = X.shape[0]
    xs = np.zeros(n)
    xs[n-1] = b[n - 1] / X[n-1, n-1]
    for i in range(n-2, -1, -1):
        xs[i] = (b[i] - np.dot(X[i, i+1:], xs[i+1:])) / X[i, i]
    return xs


X = np.array([[7, -2, 1, -1], [0, 4, -1, 3], [0, 0, 1, 2], [0, 0, 0, 5]], dtype=np.float32)
b = np.array([8, 6, 0, -3], dtype=np.float32)
sol = back_substitution(X, b)

print(sol)
