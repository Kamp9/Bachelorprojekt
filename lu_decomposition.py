import numpy as np


def lu(A):
    (m, n) = A.shape
    for k in range(m):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - A[k+1:, k] * A[k, k+1:]
    return A

