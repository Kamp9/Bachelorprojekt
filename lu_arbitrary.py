# coding=utf-8
import numpy as np
import tests
import scipy.linalg as sp


"""
def lu_block(A, r):
    m, n = A.shape
    U = np.zeros((m, m))
    L = np.zeros((m, m))
    A = A.astype(np.float64)
    for k in range(0, m, r):
        L[k:, k:k+r] = lu_out_of_place(A[k:, k:k+r])[0]
        U[k:k+r, k:] = lu_out_of_place(A[k:k+r, k:])[1]
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return L, U

def lu_partial_pivot(A):
    m, n = A.shape
    L = np.identity(m)
    U = A.astype(np.float64)
    P = range(m)
    Q = range(m)
    for k in range(m - 1):
        i = k + find_pivot(U[k:, k], 0)
        permute(P, L, U, Q, k, (i, k))
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, np.triu(U)


"""


def lu_out_of_place(A):
    A = A.astype(np.float64)
    m, n = A.shape
    if m > n:
        L = np.eye(m, n)
        U = np.zeros((n, n))
    else:
        L = np.eye(m, m)
        U = np.zeros((m, n))
    for k in range(min(m, n)):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = A[k+1:, k] / U[k, k]
        A[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return L, U


def row_substitution(U, B):
    m, n = U.shape
    r, n = B.shape
    U = U.astype(np.float64)
    B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(U[k, :k], x[:k])) / U[k, k]
    return x


def lu_block(A, r):
    m, n = A.shape
    U = np.zeros((m, m))
    L = np.zeros((m, m))
    A = A.astype(np.float64)
    for k in range(0, m, r):
        L[k:, k:k+r] = lu_out_of_place(A[k:, k:k+r])[0]
        U[k:k+r, k:k+r] = lu_out_of_place(A[k:k+r, k:k+r])[1]
        U[k:k+r, k+r:] = row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return L, U


def find_pivot(A, pivoting):
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


def permute(P, L, U, Q, k, (i, j)):
    # Permuter raekker
    if i != k:
        U[[i, k], k:] = U[[k, i], k:]
        L[[i, k], :k] = L[[k, i], :k]
        P[i], P[k] = P[k], P[i]
    # Permuter soejler
    if j != k:
        U[:, [j, k]] = U[:, [k, j]]
        Q[i], Q[k] = Q[k], Q[i]


def lu_partial_pivot(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    L = np.identity(m)
    U = np.zeros((m, m))
    P = range(m)
    Q = range(m)
    for k in range(0, m, r):
        i = k + find_pivot(U[k:, k], 0)
        permute(P, L, U, Q, k, (i, k))
        L[k:, k:k+r] = lu_out_of_place(A[k:, k:k+r])[0]
        U[k:k+r, k:] = lu_out_of_place(A[k:k+r, k:])[1]
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return P, L, np.triu(U)

rand_int_matrix = np.random.randint(-1000, 1000, size=(4, 4))


"""
a_sym = tests.generate_pos_dif(4, -1000, 1000)


print sp.lu(a_sym)[1]
print sp.lu(a_sym)[2]
print
print lu_partial_pivot(a_sym, 2)[1]
print lu_partial_pivot(a_sym, 2)[2]
"""