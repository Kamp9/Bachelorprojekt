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
            _swap_row_lower(L, m, k, x)
            _swap_row_upper(U, m, k, x)
            _swap_row_upper(P, m, 0, x)
        if k != y:
            _swap_col_lower(L, m, k, y)
            _swap_col_upper(U, m, k, y)
            _swap_col_upper(Q, m, 0, y)


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


def _maxpos(A):
    return np.unravel_index(np.argmax(np.abs(A)), A.shape)


def lu_complete_pivot(A):
    m, n = A.shape
    L = np.identity(m)
    P = np.identity(m)
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _maxpos(U[k:, k:])
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


def _maxpos_rook(A):
    A = np.abs(A)
    rowindex = A.argmax(axis=0)[0]
    colmax = A[rowindex][0]
    rowmax = 0.0
    while rowmax < colmax:  # colindex kan muligvis ikke blive tildelt hvis colmax == 0.0
        colindex = A.argmax(axis=1)[rowindex]
        rowmax = A[rowindex][colindex]
        if colmax < rowmax:
            rowindex = A.argmax(axis=0)[colindex]
            colmax = A[rowindex][colindex]
        else:
            break
    return rowindex, colindex

"""
def _maxpos_rook2(A):
    A = np.abs(A)
    colindex = A.argmax(axis=1)[0]
    rowmax = A[colindex][0]
    colmax = 0.0
    print colindex, rowmax
    while colmax < rowmax:  # colindex kan muligvis ikke blive tildelt hvis colmax == 0.0
        rowindex = A.argmax(axis=0)[colindex]
        colmax = A[rowindex][colindex]
        if rowmax < colmax:
            colindex = A.argmax(axis=1)[rowindex]
            rowmax = A[colindex][rowindex]
        else:
            break
    return rowindex, colindex
"""


def lu_rook_pivot(A):
    m, n = A.shape
    L = np.identity(m)
    P = np.identity(m)
    Q = np.identity(m)
    U = A.astype(np.float64)
    for k in range(m):
        x, y = _maxpos_rook(U[k:, k:])
        x, y = x + k, y + k
        _permute((P, Q), L, U, m, k, (x, y), 1)
        print U
        print L
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, Q, L, np.triu(U)

