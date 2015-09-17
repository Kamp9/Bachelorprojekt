# coding=utf-8
import numpy as np


def lu(A):
    (m, n) = A.shape
    L = np.zeros((m, m))
    U = np.zeros((m, m))
    for i in range(m):
        U[i, i] = A[0, 0]  # kan muligvis gøres kortere
        L[i, i] = 1        # skal måske ikke være med
        U[i, i+1:] = A[0, 1:]
        print A
        L[i+1:, i] = (1.0 / A[0, 0]) * A[1:, 0]
        A = A[1:, 1:] - np.resize(L[i+1:, 0], (i+1, 1)) * U[0, i+1:]
    return L, U

