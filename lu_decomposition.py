# coding=utf-8
import numpy as np


def lu_inplace(A):
    (m, n) = A.shape
    A = A.astype(np.float64)
    for k in range(m-1):
        A[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - A[k+1:, k, np.newaxis] * A[k, k+1:]
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    return L, np.triu(A)


def lu_out_of_place(A):
    (m, n) = A.shape
    A = A.astype(np.float64)
    U = np.zeros((m, m))
    L = np.zeros((m, m))
    np.fill_diagonal(L, 1)
    for k in range(m):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return L, U

