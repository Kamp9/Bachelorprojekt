import numpy as np
import scipy.linalg as sp
import lu_arbitrary2
from memory_profiler import profile


@profile
def cholesky(A):
    A = A.astype(np.float64)
    m, n = A.shape
    U = np.zeros((m, m))
    for k in xrange(m):
        pivot = np.math.sqrt(A[k, k])
        U[k, k] = pivot
        U[k, k+1:] = A[k, k+1:] / pivot
        A[k+1:, k+1:] -= U[k, k+1:] * U[k, k+1:, np.newaxis]
    return U


@profile
def cholesky_in_place(A):
    U = A.astype(np.float64)
    m, n = A.shape
    for k in xrange(m):
        U[k, k] = np.math.sqrt(U[k, k])
        U[k, k+1:] = U[k, k+1:] / U[k, k]
        U[k+1:, k+1:] -= U[k, k+1:] * U[k, k+1:, np.newaxis]
    return np.triu(U)


@profile
def lu_partial_block2(A, r):
    m, n = A.shape
    A = A.astype(np.float64)
    P = range(m)
    L = np.identity(m)
    U = np.zeros((m, m))
    for k in range(0, min(m, n), r):
        PLU = lu_arbitrary2.lu_partial(A[k:, k:k+r])
        temp_P = PLU[0]
        temp_P_i = lu_arbitrary2.invert_permutation_array(temp_P)
        P[k:] = lu_arbitrary2.permute_array(temp_P, P[k:])
        L[k:, k:k+r] = PLU[1]
        U[k:k+r, k:k+r] = PLU[2][:r, :r]
        L[k:, :k] = lu_arbitrary2.permute_rows(temp_P_i, L[k:, :k])
        A[k:, k:] = lu_arbitrary2.permute_rows(temp_P_i, A[k:, k:])
        U[k:k+r, k+r:] = lu_arbitrary2.row_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(L[k+r:, k:k+r], U[k:k+r, k+r:])
    return lu_arbitrary2.P_to_Pmatrix(P), L, U

a = np.random.random_integers(-1000, 1000, size=(1000, 1000))
b = np.random.random_integers(1000000, 100000000, size=(1000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)

cholesky(a_sym)
cholesky_in_place(a_sym)

# sp.cholesky(a_sym)

#lu_partial_block2(a, 42)
