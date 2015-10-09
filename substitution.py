# coding=utf-8
import numpy as np
import lu_decomposition


def _forward_substitution(L, b):
    (m, n) = L.shape
    z = np.zeros(m)
    for k in range(m):
        z[k] = (1.0 / L[k, k]) * (b[k] - np.dot(L[k, :k], z[:k]))
    z = z[:, np.newaxis]
    return z


def _back_substitution(U, z):
    (m, n) = U.shape
    l = m - 1
    x = np.zeros(m)
    for k in range(m):
        x[l-k] = (1.0 / U[l-k, l-k]) * (z[l-k] - np.dot(U[l-k, l-k:], x[l-k:]))
    x = x[:, np.newaxis]
    return x


def solve(A, b, pivoting):
    # No pivoting
    if pivoting == 0:
        L, U = lu_decomposition.lu_out_of_place(A)
        z = _forward_substitution(L, b)
        x = _back_substitution(U, z)


def complete_solve(A, b):
    P, Q, L, U = lu_complete_pivot(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def out_of_place_solve(A, b):
    L, U = lu_out_of_place(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x
