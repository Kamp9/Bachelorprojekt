# coding=utf-8
import numpy as np


def lu_in_place(A):
    m, n = A.shape
    U = A.astype(np.float64)
    for k in range(m-1):
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] -= U[k+1:, k, np.newaxis] * U[k, k+1:]
    L = np.tril(U)
    np.fill_diagonal(L, 1)
    return L, np.triu(U)


def lu_inplace_with_dot(A):
    m, n = A.shape
    A = A.astype(np.float64)
    for k in range(m-1):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] -= np.dot(A[k+1:, k, np.newaxis], A[np.newaxis, k, k+1:])
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
        L[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return L, U


def find_pivot(A, pivoting):
    A = np.abs(A)

    # Partial Pivot
    if pivoting == 0:
        return A.argmax()

    # Complete Pivot
    if pivoting == 1:
        return np.unravel_index(np.argmax(A), A.shape)

    # Rook Pivot
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
    for k in range(m - 1):
        pivot = k + find_pivot(U[k:, k], 0)
        _permute(P, L, U, m, k, pivot, 0)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), L, np.triu(U)


def lu_complete_pivot(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    for k in range(m - 1):
        x, y = find_pivot(U[k:, k:], 1)
        x, y = x + k, y + k
        _permute((P, Q), L, U, m, k, (x, y), 1)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), Q.transpose(), L, np.triu(U)


def lu_rook_pivot(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = np.identity(m)
    Q = np.identity(m)
    L = np.identity(m)
    for k in range(m - 1):
        x, y = find_pivot(U[k:, k:], 2)
        x, y = x + k, y + k
        _permute((P, Q), L, U, m, k, (x, y), 1)
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P.transpose(), Q.transpose(), L, np.triu(U)


def row_sub(L, B):
    m, n = L.shape
    r, n = B.shape
    L = L.astype(np.float64)
    B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(L[k, :k], x[:k]))
    return x


def col_sub(U, B):
    m, n = U.shape
    n, r = B.shape
    U = U.astype(np.float64)
    B = B.astype(np.float64)
    x = np.zeros((n, r))
    for k in range(m):
        x[:, k] = (B[:, k] - np.dot(x[:, :k], U[:k, k])) / U[k, k]
    return x


def lu_block(A, r):
    m, n = A.shape
    U = np.zeros((m, m))
    L = np.zeros((m, m))
    A = A.astype(np.float64)
    for k in range(0, m, r):
        decomp = lu_in_place(A[k:k+r, k:k+r])
        L[k:k+r, k:k+r] = decomp[0]
        U[k:k+r, k:k+r] = decomp[1]
        L[k+r:, k:k+r] = col_sub(U[k:k+r, k:k+r], A[k+r:, k:k+r])
        U[k:k+r, k+r:] = row_sub(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return L, U
