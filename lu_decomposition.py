# coding=utf-8
import numpy as np
import substitution


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
    L = np.identity(m)
    for k in range(m):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return L, U


def solve(A, b):
    L, U = lu_inplace(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def lu_partial_pivot(A):
    (m, n) = A.shape
    L = np.identity(m)
    P = np.identity(m)  # kan være P = L
    U = A.astype(np.float64)
    for k in range(m):
        pivot = k + np.absolute(U[k:, k]).argmax(axis=0)
        if k != pivot:
            temp = U[k, :].copy()  # kan vi undgå copy()?
            U[k, :] = U[pivot, :]
            U[pivot, :] = temp
            temp = P[k, :].copy()  # kan vi undgå copy()?
            P[k, :] = P[pivot, :]
            P[pivot, :] = temp
            if k >= 1:
                temp = L[k, :k].copy()  # kan vi undgå copy()?
                L[k, :k] = L[pivot, :k]
                L[pivot, :k] = temp
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, np.triu(U)


def _find_max(A):
    (m, n) = A.shape
    max_entry = (A[0, 0], 0, 0)
    for i in range(m):
        for j in range(n):
            if A[i, j] > max_entry[0]:
                max_entry = (A[i, j], i, j)
    return max_entry[1], max_entry[2]


def lu_full_pivot(A):
    (m, n) = A.shape
    L = np.identity(m)
    P = np.identity(m)  # kan være P = L
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _find_max(A[k:, k:])
        x, y = (x + k, y + k)
        if (k, k) != pivot:
            temp = U[k, :].copy()  # kan vi undgå copy()?
            U[k, :] = U[pivot, :]
            U[pivot, :] = temp
            temp = P[k, :].copy()  # kan vi undgå copy()?
            P[k, :] = P[pivot, :]
            P[pivot, :] = temp
            if k >= 1:
                temp = L[k, :k].copy()  # kan vi undgå copy()?
                L[k, :k] = L[pivot, :k]
                L[pivot, :k] = temp
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, np.triu(U)