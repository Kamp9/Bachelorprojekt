# coding=utf-8
import numpy as np


def is_pos_def(A):
    return all(np.linalg.eigvals(A) > 0)


def cholesky(A):
    if is_pos_def(A):
        (m, n) = A.shape
        L = np.zeros((m, m))
        for i in range(m):
            a00 = np.math.sqrt(A[0, 0])
            L[i, i] = a00
            if m != 1:
                L10 = A[1:, 0] / a00
                L[i+1:, i] = L10
                A = A[1:, 1:] - L10[:, np.newaxis] * L10[np.newaxis, :]
                (m, n) = A.shape
        return L
    else:
        raise ValueError('Matrix is not positive definite')


def forward_substitution(L, b):
    (m, n) = L.shape
    z = np.zeros(m)
    for i in range(m):
        z[i] = (1.0 / L[i, i]) * (b[i] - np.dot(L[i, :i], z[:i]))
    return z


def backward_substitution(U, z):
    (m, n) = U.shape
    n = m - 1       # kan gøres pænere
    x = np.zeros(m)
    for i in range(m):
        x[n-i] = (1.0 / U[n-i, n-i]) * (z[n-i] - np.dot(U[n-i, :n-i], x[:n-i]))
    return x









