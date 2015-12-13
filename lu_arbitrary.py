# coding=utf-8
import numpy as np
import tests
import scipy.linalg as sp
np.set_printoptions(linewidth=200)

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

def lu_partial_block(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    L = np.identity(m)
    U = np.zeros((m, m))
    P = range(m)
    Q = range(m)
    for k in range(0, m, r):
        L[k:, k:k+r] = lu_out_of_place(A[k:, k:k+r])[0]
        U[k:k+r, k:] = lu_out_of_place(A[k:k+r, k:])[1]
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
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


def row_substitution(L, B):
    m, n = L.shape
    r, n = B.shape
    L = L.astype(np.float64)
    B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(L[k, :k], x[:k])) / L[k, k]
    return x


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


def permute_partial(P, L, A, k, i):
    # Permuter raekker
    if i != k:
        P[i], P[k] = P[k], P[i]
        L[[i, k], :k] = L[[k, i], :k]
        A[[i, k], k:] = A[[k, i], k:]


def lu_partial(A):
    A = A.astype(np.float64)
    m, n = A.shape
    P = range(m)
    if m > n:
        L = np.eye(m, n)
        U = np.zeros((n, n))
    else:
        L = np.eye(m, m)
        U = np.zeros((m, n))
    for k in range(min(m, n)):
        i = k + find_pivot(A[k:, k], 0)
        permute_partial(P, L, A, k, i)
        U[k, k:] = A[k, k:]
        L[k+1:, k] = A[k+1:, k] / U[k, k]
        A[k+1:, k+1:] -= L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, U


def permute_rows(P, A):
    PA = np.empty(A.shape)
    for i in range(len(P)):
        PA[P[i], :] = A[i, :]
    return PA


def transpose_array(P):
    P_transpose = np.zeros(len(P), dtype=int)
    for i, e in enumerate(P):
        P_transpose[e] = i
    return P_transpose


def P_to_Pmatrix(P):
    m = len(P)
    Pmatrix = np.zeros((m, m))
    for i, e in enumerate(P):
        Pmatrix[e, i] = 1
    return Pmatrix


def permute_array(P, a):
    Pa = np.zeros(len(P), dtype=int)
    for i, e in enumerate(P):
        Pa[i] = a[P[i]]
    return Pa


def lu_partial_block(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    L = np.identity(m)
    U = np.zeros((m, m))
    Plal = np.identity(m)
    for k in range(0, m, r):
        PLU = lu_partial(A[k:, k:k+r])
        P = P_to_Pmatrix(PLU[0])
        L[k:, k:k+r] = PLU[1]
        U[k:k+r, k:k+r] = PLU[2][:r, :r]
        L[k:, :k] = np.dot(P.transpose(), L[k:, :k])
        A[k:, k:] = np.dot(P.transpose(), A[k:, k:])
        Plal[:, k:] = np.dot(Plal[:, k:], P)
        U[k:k+r, k+r:] = row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return Plal, L, U


def lu_partial_block2(A, r):
    """
    Skal nok også virke for ikke kvadratiske matricer
    :param A:
    :param r:
    :return:
    """
    m, n = A.shape
    A = A.astype(np.float64)
    P = range(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(0, m, r):
        PLU = lu_partial(A[k:, k:k+r])
        small_P = PLU[0]
        small_P_t = transpose_array(small_P)
        P[k:] = permute_array(small_P, P[k:])
        L[k:, k:k+r] = PLU[1]
        U[k:k+r, k:k+r] = PLU[2][:r, :r]
        L[k:, :k] = permute_rows(small_P_t, L[k:, :k])
        A[k:, k:] = permute_rows(small_P_t, A[k:, k:])
        U[k:k+r, k+r:] = row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return P_to_Pmatrix(P), L, U


rand_int_matrix = np.random.randint(-1000, 1000, size=(6, 6))
a_sym = tests.generate_pos_dif(4, -1000, 1000)
#print rand_int_matrix

matrix = np.array([[-874, -965, 18, -71],
                   [230, -457, -817, -508],
                   [570, -781, -109, -751],
                   [-4, -497, -630, 230]])

matrix2 = np.array([[-549, -257, -184, 661],
                    [903, 272, -341, -451],
                    [628, 484, -577, 475],
                    [-42, -474, 935, -423]])

a = tests.generate_pos_dif(4, -1000, 1000)
