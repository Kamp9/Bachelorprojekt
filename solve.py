# coding=utf-8
import numpy as np
import lu_decomposition


def _forward_substitution(L, b):
    (m, n) = L.shape
    z = np.zeros(m)
    for k in range(m):
        z[k] = (1.0 / L[k, k]) * (b[k] - np.dot(L[k, :k], z[:k]))
    return z[:, np.newaxis]


def _back_substitution(U, z):
    (m, n) = U.shape
    l = m - 1
    x = np.zeros(m)
    for k in range(m):
        x[l-k] = (1.0 / U[l-k, l-k]) * (z[l-k] - np.dot(U[l-k, l-k:], x[l-k:]))
    return x[:, np.newaxis]


def solve(A, b, pivoting):
    # Skal gerne returnere sammen dimentioner som b, hvis den skal være ligesom scipy
    # No pivoting
    if pivoting == 0:
        L, U = lu_decomposition.lu_out_of_place(A)
        z = _forward_substitution(L, b)
        x = _back_substitution(U, z)
        return x

    # Partial pivoting
    if pivoting == 1:
        P, L, U = lu_decomposition.lu_partial_pivot(A)
        z = _forward_substitution(L, np.dot(P.transpose(), b))
        x = _back_substitution(U, z)
        return x

    # Complete pivoting
    if pivoting == 2:
        P, Q, L, U = lu_decomposition.lu_complete_pivot(A)
        z = _forward_substitution(L, np.dot(P, b))
        x = np.dot(_back_substitution(U, z), Q)
        return x

    # Rook pivoting
    if pivoting == 3:
        P, Q, L, U = lu_decomposition.lu_rook_pivot(A)
        z = _forward_substitution(L, np.dot(P, b))
        x = np.dot(_back_substitution(U, z), Q)
        return x
