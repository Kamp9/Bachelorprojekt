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


def _swap_row(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[:] = A[k, :]
    A[k, :] = A[pivot, :]
    A[pivot, :] = temp[:]


def _swap_row_L(A, m, k, pivot):  # Skal vi virkelig have den her?
    temp = np.empty(m)
    temp[:k] = A[k, :k]
    A[k, :k] = A[pivot, :k]
    A[pivot, :k] = temp[:k]


def _swap_col(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[:] = A[:, k]
    A[:, k] = A[:, pivot]
    A[:, pivot] = temp[:]


def _swap_col_U(A, m, k, pivot):  # kan være ikke at tage m med videre er ligeså hurtigt
    temp = np.empty(m)
    temp[:k] = A[:k, k]
    A[:k, k] = A[:k, pivot]
    A[:k, pivot] = temp[:k]


def _permute(P, A, L, U, m, k, pivot, pivoting):  # permute skal nok regne m ud via shape
    if pivoting == 0:
        if k != pivot:
            _swap_row(P, m, k, pivot)
            _swap_row(A, m, k, pivot)
            _swap_row_L(L, m, k, pivot)  # Ihh altså

    if pivoting == 1:
        P, Q = P
        x, y = pivot
        if k != x:
            _swap_row(A, m, k, x)
            _swap_row(P, m, k, x)
            _swap_row_L(L, m, k, x)
        if k != y:
            _swap_col(A, m, k, y)
            _swap_col(Q, m, k, y)
            _swap_col_U(U, m, k, y)


def lu_partial_pivot(A):
    m, n = A.shape
    A = A.astype(np.float64)
    P = np.identity(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(m):
        pivot = k + np.abs(A[k:, k]).argmax(axis=0)
        _permute(P, A, L, P, m, k, pivot, 0)
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, U


def _maxpos(A):
    return np.unravel_index(np.argmax(np.abs(A)), A.shape)


def lu_complete_pivot(A):
    m, n = A.shape
    A = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(m):
        x, y = _maxpos(A[k:, k:])
        x, y = x + k, y + k
        _permute((P, Q), A, L, U, m, k, (x, y), 1)
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, Q, L, U


def complete_solve(A, b):
    P, Q, L, U = lu_complete_pivot(A)  # kan også være lu_out_of_place(A)
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def _maxpos_rook(A):
    A = np.abs(A)
    rowindex = A.argmax(axis=0)[0]
    colmax = A[rowindex][0]
    rowmax = -1
    while rowmax < colmax:
        colindex = A.argmax(axis=1)[rowindex]
        rowmax = A[rowindex][colindex]
        if colmax < rowmax:
            rowindex = A.argmax(axis=0)[colindex]
            colmax = A[rowindex][colindex]
        else:
            break
    return rowindex, colindex


def lu_rook_pivot(A):
    m, n = A.shape
    A = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(m):
        x, y = _maxpos_rook(A[k:, k:])
        x, y = x + k, y + k
        _permute((P, Q), A, L, U, m, k, (x, y), 1)
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, Q, L, U

