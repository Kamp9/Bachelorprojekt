# coding=utf-8
import numpy as np


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


def _find_pivot(A, pivoting):
    A = np.abs(A)

    # partial pivoting
    if pivoting == 0:
        return A.argmax()

    # complete pivoting
    if pivoting == 1:
        return np.unravel_index(np.argmax(A), A.shape)

    # rook pivoting
    if pivoting == 2:
        rowindex = A[:, 0].argmax()
        colmax = A[rowindex, 0]
        rowmax = -1
        while rowmax < colmax:
            colindex = A[rowindex].argmax()
            rowmax = A[rowindex][colindex]
            if colmax < rowmax:
                rowindex = A[:, colindex].argmax()
                colmax = A[rowindex][colindex]
            else:
                break
        return rowindex, colindex


def _swap_row(A, m, k, pivot):
    temp = np.empty(m)
    temp[:] = A[k, :]
    A[k, :] = A[pivot, :]
    A[pivot, :] = temp[:]


def _swap_row_to_k(A, m, k, pivot):
    temp = np.empty(m)
    temp[:k] = A[k, :k]
    A[k, :k] = A[pivot, :k]
    A[pivot, :k] = temp[:k]


def _swap_col(A, m, k, pivot):
    temp = np.empty(m)
    temp[:] = A[:, k]
    A[:, k] = A[:, pivot]
    A[:, pivot] = temp[:]


def _permute(P, L, U, m, k, pivot, pivoting):  # permute skal nok regne m ud via shape
    # One dimensional pivoting
    if pivoting == 0:
        if k != pivot:
            _swap_row(P, m, k, pivot)
            _swap_row(U, m, k, pivot)
            _swap_row_to_k(L, m, k, pivot)

    # Two dimensional pivoting
    if pivoting == 1:
        P, Q = P
        x, y = pivot
        if k != x:
            _swap_row(U, m, k, x)
            _swap_row(P, m, k, x)
            _swap_row_to_k(L, m, k, x)
        if k != y:
            _swap_col(U, m, k, y)
            _swap_col(Q, m, k, y)


def lu_partial_pivot(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = np.identity(m)
    L = np.identity(m)
    for k in range(m):
        pivot = k + _find_pivot(U[k:, k], 0)
        _permute(P, L, U, m, k, pivot, 0)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), L, np.triu(U)


def lu_complete_pivot(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    for k in range(m):
        i, j = _find_pivot(U[k:, k:], 1)
        i, j = i + k, j + k
        _permute((P, Q), L, U, m, k, (i, j), 1)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), Q.transpose(), L, np.triu(U)


def lu_rook_pivot(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    for k in range(m):
        i, j = _find_pivot(U[k:, k:], 2)
        i, j = i + k, j + k
        _permute((P, Q), L, U, m, k, (i, j), 1)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), Q.transpose(), L, np.triu(U)
