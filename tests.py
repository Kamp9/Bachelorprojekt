import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import lu_decomposition2
import solve
import naive_stolen_from_web
from numpy.testing import TestCase, assert_array_almost_equal
from decimal import Decimal


real_matrix = np.array([[3.3821, 0.8784, 0.3613, -2.0349],
                        [0.8784, 2.0068, 0.5587, 32.1169],
                        [0.3613, 0.5587, 3.6656, 0.7807],
                        [-100.0349, 0.1169, 11.7807, 2.5397]])

int_matrix2 = np.array([[1, 3, 5],
                        [2, 4, 7],
                        [1, 1, 0]])

int_matrix3 = np.array([[8, 2, 9],
                        [4, 9, 4],
                        [6, 7, 9]])

int_matrix4 = np.array([[0, 1],
                        [1, 2]])

imag_matrix = np.array([[1.+0.j, 0.-2.j],
                        [0.+2.j, 5.+0.j]])

rand_matrix = np.random.random((2, 2))

matrix1 = np.array([[0.5],
                    [0.75]])

matrix2 = np.array([2, 9])

A = np.array([[9, 4],
              [7, 9]])


rand_matrix2 = np.random.rand(1000, 1000)
rand_col = np.random.rand(1000, 1)

lalala = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rooktest = np.array([[155, 53113, 642, 531],
                     [-4, 553, 700, 1],
                     [-90053, 53534, 513, 35],
                     [-53353, 53, 1000, -353]])


b3 = np.array([[41],
               [31],
               [64]])

a = np.array([[5315353, -5313451, 65153511], [-1355351, -35315, 35362], [0.00005353135, -5, 13646]])
b4 = np.array([4133, 6421, -533, -5533])


P, Q, L, U = lu.lu_rook_pivot(rooktest)

P2, L2, U2, Q2, n = lu_decomposition2.lu_decompose(rooktest, 0)

"""
print P2
print Q2
print L2
print U2
print n

print P2, np.dot(L2, U2)

print solve.solve_cholesky(posdef_matrix, b3)

print lu_decomposition.lu_partial_pivot(rooktest)[0]

rand_int_matrix = np.random.randint(-1000, 1000, size=(10, 10))
rand_int_col = np.random.randint(-1000, 1000, size=(10, 1))

print solve.solve(rand_int_matrix, rand_int_col, 1)

print Decimal(solve.solve(rand_int_matrix, rand_int_col, 1)[0][0])


print Decimal(0.1)
"""

lala = np.array(([3, 8, 9],
                 [4, 5, 6],
                 [2, 5, 3]))

lal = np.array([[1, 0, 0],
                [-42, 1, 0],
                [1, -31, 1]])

lal2 = np.array(([100, 42],
                 [42, 143]))

lala2 = np.array(([0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]))

lala3 = np.array(([1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]))

posdef_matrix = np.array([[1000, -1, 63],
                          [-1, 2042, -552],
                          [63, -552, 5535]])

a = np.random.random_integers(-1000, 1000, size=(6, 6))
b = np.random.random_integers(10000, 1000000, size=(6, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)

block = np.array([[-1, 531, 53],
                  [-42, 42, 531],
                  [1, -3531, -13],
                  [-1, 214, 61],
                  [-42, 2, 315],
                  [1, -31, -13],
                  [-1, 351, 1],
                  [-42, 1, 53],
                  [1, -31, -13],
                  [-1, 631, 141],
                  [-42, 2, 31],
                  [1, -31, -13]])



small_block = np.array(([1, 1, 1],
                        [0, 1, 1],
                        [0, 0, 1]))


lal2 = np.array([[-115, 42, 315],
                 [0, 393, -131],
                 [0, 0, 515]])

# print cholesky.cholesky(a_sym)


def forward_substitution(U, B):
    m, n = U.shape
    r, n = B.shape
    U = U.astype(np.float64)
    B = B.astype(np.float64)
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(U[k, :k], x[:k])) / U[k, k]
    return x


def back_substitution(U, B):
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
        decomp = lu.lu_inplace(A[k:k+r, k:k+r])
        L[k:k+r, k:k+r] = decomp[0]
        U[k:k+r, k:k+r] = decomp[1]
        L[k:k+r, k+r:] = forward_substitution(L[k:k+r, k:k+r], A[k:k+r, k+r:])
        U[k+r:, k:k+r] = back_substitution(U[k:k+r, k:k+r], A[k+r:, k:k+r])
        A[k+r:, k+r:] -= np.dot(U[k+r:, k:k+r], L[k:k+r, k+r:])
    return L, U

rand_int_matrix = np.random.randint(-1000, 1000, size=(10, 10))
print back_substitution(lu.lu_inplace(rand_int_matrix[:2, :2])[1], rand_int_matrix[2:, :2])
print lu.lu_inplace(rand_int_matrix)[0]
