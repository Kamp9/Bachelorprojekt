# coding=utf-8
import numpy as np
import tests
import lu_arbitrary
import scipy.linalg as sp
import lu
np.set_printoptions(linewidth=200)


def lu_out_of_place(A):
    m, n = A.shape
    U = A.astype(np.float64)
    for k in range(min(m, n)):
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] -= U[k+1:, k, np.newaxis] * U[k, k+1:]
    L = np.tril(U)
    np.fill_diagonal(L, 1)
    return L[:m, :m], np.triu(U[:n, :n])


def row_substitution(L, B):
    m, n = L.shape
    r, n = B.shape
    #L = L.astype(np.float64)
    #B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(L[k, :k], x[:k])) # / L[k, k]  # / L[k, k] burde kunne fjernes
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


def permute_partial(P, A, k, i):
    # Permuter raekker
    if i != k:
        P[i], P[k] = P[k], P[i]
        A[[i, k]] = A[[k, i]]


def permute_partial2(P, A, k, i):
    m, n = A.shape
    temp = np.empty(n)
    if i != k:
        P[i], P[k] = P[k], P[i]
        temp[:] = A[k, :]
        A[k, :] = A[i, :]
        A[i, :] = temp[:]

"""
    m, n = A.shape
    U = A.astype(np.float64)
    for k in range(min(m, n)):
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] -= U[k+1:, k, np.newaxis] * U[k, k+1:]
    L = np.tril(U)
    np.fill_diagonal(L, 1)
    return L[:m, :m], np.triu(U[:n, :n])
"""


def permute_rows(P, A):
    PA = np.empty(A.shape)
    for i in range(len(P)):
        PA[P[i], :] = A[i, :]
    return PA


def permute_cols(Q, A):
    AQ = np.empty(A.shape)
    for j in range(len(Q)):
        AQ[:, Q[j]] = A[:, j]
    return AQ

def invert_permutation_array(P):
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
    for i in range(len(P)):
        Pa[i] = a[P[i]]
    return Pa


def lu_partial(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = range(m)
    for k in range(min(m, n)):
        i = k + find_pivot(U[k:, k], 0)
        permute_partial(P, U, k, i)
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] -= U[k+1:, k, np.newaxis] * U[k, k+1:]
    L = np.tril(U)[:m, :m]
    np.fill_diagonal(L, 1)
    return P, L, np.triu(U[:n, :n])


def lu_partial_block2(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    P = range(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(0, min(m, n), r):
        PLU = lu_partial(A[k:, k:k+r])
        temp_P = PLU[0]
        temp_P_i = invert_permutation_array(temp_P)
        P[k:] = permute_array(temp_P, P[k:])
        L[k:, k:k+r] = PLU[1]
        U[k:k+r, k:k+r] = PLU[2][:r, :r]
        L[k:, :k] = permute_rows(temp_P_i, L[k:, :k])
        A[k:, k:] = permute_rows(temp_P_i, A[k:, k:])
        U[k:k+r, k+r:] = row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return P_to_Pmatrix(P), L, U

"""
def permute_complete(P, Q, A, k, i, j):
    # Permuter raekker
    if i != k:
        P[i], P[k] = P[k], P[i]
        A[[i, k]] = A[[k, i]]
    if j != k:
        Q[j], Q[k] = Q[k], Q[j]
        A[:, [j, k]] = A[:, [k, j]]


def lu_complete(A):
    m, n = A.shape
    U = A.astype(np.float64)
    P = range(m)
    Q = range(n)
    for k in range(min(m, n)):
        i, j = find_pivot(U[k:, k:], 1)
        i, j = i + k, j + k
        permute_complete(P, Q, U, k, i, j)
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] -= U[k+1:, k, np.newaxis] * U[k, k+1:]
    L = np.tril(U)[:m, :m]
    np.fill_diagonal(L, 1)
    return P, Q, L, np.triu(U[:n, :n])


def lu_complete_block(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    P = range(m)
    Q = range(n)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(0, min(m, n), r):
        PQLU = lu_complete(A[k:, k:k+r])
        temp_P = PQLU[0]
        temp_Q = PQLU[1]
        print temp_P
        print temp_Q
        temp_P_i = invert_permutation_array(temp_P)
        temp_Q_i = invert_permutation_array(temp_Q)
        P[k:] = permute_array(temp_P, P[k:])
        Q[k:] = permute_array(temp_Q, Q[k:])
        L[k:, k:k+r] = PQLU[2]
        U[k:k+r, k:k+r] = PQLU[3][:r, :r]
        L[k:, :k] = permute_rows(temp_P_i, L[k:, :k])
        U[:k, k:] = permute_cols(temp_Q_i, U[:k, k:])
        A[k:, k:] = permute_rows(temp_P_i, A[k:, k:])
        A[k:, k:] = permute_cols(temp_Q_i, A[k:, k:])
        U[k:k+r, k+r:] = row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return P_to_Pmatrix(P), P_to_Pmatrix(Q), L, U
"""

