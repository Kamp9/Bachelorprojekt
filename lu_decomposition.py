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


def out_of_place_solve(A, b):
    L, U = lu_out_of_place(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def in_place_solve(A, b):
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


def partial_solve(A, b):
    P, L, U = lu_partial_pivot(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def _find_max(A):
    (m, n) = A.shape
    max_entry = (A[0, 0], 0, 0)
    for i in range(m):
        for j in range(n):
            if A[i, j] > max_entry[0]:
                max_entry = (A[i, j], i, j)
    return max_entry[1], max_entry[2]


def _permutate(P, Q, L, U, k, x, y):
    if k != x:
        temp = U[k, :].copy()
        U[k, :] = U[x, :]
        U[x, :] = temp
        temp = P[k, :].copy()
        P[k, :] = P[x, :]
        P[x, :] = temp
        if k >= 1:
            temp = L[k, :k].copy()
            L[k, :k] = L[x, :k]
            L[x, :k] = temp
    if k != y:
        temp = U[:, k].copy()
        U[:, k] = U[:, y]
        U[:, y] = temp
        temp = Q[:, k].copy()
        Q[:, k] = Q[:, y]
        Q[:, y] = temp
        if k >= 1:
            temp = L[:k, k].copy()
            L[:k, k] = L[:k, y]
            L[:k, y] = temp
    return P, Q, L, U


def lu_complete_pivot(A):
    (m, n) = A.shape
    L = np.identity(m)
    P = np.identity(m)  # kan være P = L
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _find_max(A[k:, k:])
        x, y = x + k, y + k
        P, Q, L, U = _permutate(P, Q, L, U, k, x, y)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, Q, L, np.triu(U)


def complete_solve(A, b):
    P, Q, L, U = lu_complete_pivot(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


"""
def lu_full_pivot(A):
    (m, n) = A.shape
    L = np.identity(m)
    P = np.identity(m)  # kan være P = L
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _find_max(A[k:, k:])
        x, y = (x + k, y + k)
        if k != x:
            temp = U[k, :].copy()
            U[k, :] = U[x, :]
            U[x, :] = temp
            temp = P[k, :].copy()
            P[k, :] = P[x, :]
            P[x, :] = temp
            if k >= 1:
                temp = L[k, :k].copy()
                L[k, :k] = L[x, :k]
                L[x, :k] = temp
        if k != y:
            temp = U[:, k].copy()
            U[:, k] = U[:, y]
            U[:, y] = temp
            temp = Q[:, k].copy()
            Q[:, k] = Q[:, y]
            Q[:, y] = temp
            if k >= 1:
                temp = L[:k, k].copy()
                L[:k, k] = L[:k, y]
                L[:k, y] = temp
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, Q, L, np.triu(U)

"""