# coding=utf-8
import numpy as np


def lu(A):
    (m, n) = A.shape
    A = A.astype(np.float64)
    for k in range(m-1):
        A[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - A[k+1:, k, np.newaxis] * A[k, k+1:]
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    return L, np.triu(A)


