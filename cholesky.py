# coding=utf-8
import numpy as np


def is_pos_def(A):
    return all(np.linalg.eigvals(A) > 0)


def cholesky2(A):
    # if is_pos_def(A):
    A = np.array(A)
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
    # else:
    #    raise ValueError('Matrix is not positive definite')


def cholesky(A):
    A = A.astype(np.float64)
    # if not is_pos_def(A):
    #    raise ValueError('Matrix is not positive definite')
    # else:
    m, n = A.shape
    U = np.zeros((m, m))
    for k in range(m):
        pivot = np.math.sqrt(A[k, k])
        U[k, k] = pivot
        U[k, k+1:] = A[k, k+1:] / pivot
        A[k+1:, k+1:] -= U[k, k+1:] * U[k, k+1:, np.newaxis]
    return U


def forward_substitution(U, B):
    m, n = U.shape
    r, n = B.shape
    # U = U.astype(np.float64)
    # B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(U[k, :k], x[:k])) / U[k, k]
    return x


def cholesky_block(A, r):
    m, n = A.shape
    U = np.zeros((m, m))
    A = A.astype(np.float64)
    # r skal gå op i m
    for k in range(0, m, r):
        U[k:k+r, k:k+r] = cholesky(A[k:k+r, k:k+r])
        U[k:k+r, k+r:] = forward_substitution(U[k:k+r, k:k+r].transpose(), A[k:k+r, k+r:])
        A[k+r:, k+r:] -= (np.dot(U[k:k+r, k+r:].transpose(), U[k:k+r, k+r:]))
    return U


