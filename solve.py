# coding=utf-8
import numpy as np
import lu
import cholesky


def forward_substitution(L, b):
    m, n = L.shape
    z = np.zeros(m)
    for k in range(m):
        z[k] = b[k] - np.dot(L[k, :k], z[:k])
    return z[:, np.newaxis]


def back_substitution(U, z):
    """
    Burde laves uden l-k men bare med omvendt range(m)
    :param U:
    :param z:
    :return:
    """
    m, n = U.shape
    l = m - 1
    x = np.zeros(m)
    for k in range(m):
        x[l-k] = (1.0 / U[l-k, l-k]) * (z[l-k] - np.dot(U[l-k, l-k:], x[l-k:]))
    return x[:, np.newaxis]


def solve_cholesky(A, b):
    # virker ikke på grund af at (1.0 / L[k, k]) *  er fjernet.
    L = cholesky.cholesky_out_of_place(A)
    U = L.transpose()
    z = forward_substitution(L, b)
    x = back_substitution(U, z)
    return x


def solve(A, b, pivoting):
    # Skal gerne returnere sammen dimentioner som b, hvis den skal være ligesom scipy
    # No pivoting
    if pivoting == 0:
        L, U = lu.lu_out_of_place(A)
        z = forward_substitution(L, b)
        x = back_substitution(U, z)
        return x

    # Partial pivoting
    if pivoting == 1:
        P, L, U = lu.lu_partial_pivot(A)
        z = forward_substitution(L, np.dot(P.transpose(), b))
        x = back_substitution(U, z)
        return x

    # Complete pivoting
    if pivoting == 2:
        P, Q, L, U = lu.lu_complete_pivot(A)
        z = forward_substitution(L, np.dot(P.transpose(), b))
        x = np.dot(Q.transpose(), back_substitution(U, z))
        return x

    # Rook pivoting
    if pivoting == 3:
        P, Q, L, U = lu.lu_rook_pivot(A)
        z = forward_substitution(L, np.dot(P.transpose(), b))
        x = np.dot(Q.transpose(), back_substitution(U, z))
        return x


def inverse(A):
    (m, n) = A.shape
    L, U = lu.lu_in_place(A)
    A_inverse = np.zeros((m, m))
    b = np.identity(m)
    for k in range(m):
        z = forward_substitution(L, b[:, k])
        x = back_substitution(U, z)
        A_inverse[:, k, np.newaxis] = x
    return A_inverse

