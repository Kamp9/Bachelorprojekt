# coding=utf-8
import numpy as np
import substitution


def lu_inplace(A):
    m, n = A.shape
    A = A.astype(np.float64)
    for k in range(m-1):
        A[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - A[k+1:, k, np.newaxis] * A[k, k+1:]
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    return L, np.triu(A)


def lu_out_of_place(A):
    m, n = A.shape
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


def _swap_row_upper(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[k:] = A[k, k:]
    A[k, k:] = A[pivot, k:]
    A[pivot, k:] = temp[k:]


def _swap_row_lower(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[:k] = A[k, :k]
    A[k, :k] = A[pivot, :k]
    A[pivot, :k] = temp[:k]


def _swap_col_upper(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[k:] = A[k:, k]
    A[k:, k] = A[k:, pivot]
    A[k:, pivot] = temp[k:]


def _swap_col_lower(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[:k] = A[:k, k]
    A[:k, k] = A[:k, pivot]
    A[:k, pivot] = temp[:k]


def _permute(P, L, U, m, k, pivot, pivoting):  # permute skal nok regne m ud via shape
    if pivoting == 0:
        if k != pivot:
            _swap_row_lower(L, m, k, pivot)
            _swap_row_upper(U, m, k, pivot)
            _swap_row_upper(P, m, 0, pivot)
    if pivoting == 1:
        P, Q = P
        x, y = pivot
        if k != x:
            _swap_row_upper(U, m, k, x)
            _swap_row_upper(P, m, 0, x)
            _swap_row_lower(L, m, k, x)
        if k != y:
            _swap_col_upper(U, m, k, y)
            _swap_col_upper(Q, m, 0, y)
            _swap_col_lower(L, m, k, y)
"""
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
"""


def lu_partial_pivot(A):
    m, n = A.shape
    L = np.identity(m)
    P = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        pivot = k + np.absolute(U[k:, k]).argmax(axis=0)
        _permute(P, L, U, m, k, pivot, 0)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, np.triu(U)


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


def _maxpos(a):
    return np.unravel_index(np.argmax(np.abs(a)), a.shape)


def lu_complete_pivot(A):
    m, n = A.shape
    L = np.identity(m)
    P = np.identity(m)
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _maxpos(A[k:, k:])
        x, y = x + k, y + k
        _permute((P, Q), L, U, m, k, (x, y), 1)
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