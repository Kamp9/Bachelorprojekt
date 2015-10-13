import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
import solve
from numpy.testing import TestCase, assert_array_almost_equal

"""
profiling
"""

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

#  (P, L, U) = sp.linalg.lu(int_matrix2)

bb = np.array([[7],
               [-4.153],
               [43]])

fun = np.array([[0.5],
                [0.75]])

fun2 = np.array([[2, 9]])

fun3 = np.array([[0.5, 0.75]])

a = np.array([[8, 2, 3], [2, 9, 3], [3, 3, 6]])
l = cholesky_decomposition.cholesky(a)
lt = l.transpose()

rand_matrix2 = np.random.rand(15, 15)
rand_col = np.random.rand(15, 1)

lalala = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rooktest = np.array([[155, -800, 642],
                     [-4, 553, 700],
                     [-900, 732, 513]])

a = np.array([[5315353, -5313451, 65153511], [-1355351, -35315, 35362], [0.00005353135, -5, 13646]])
b = np.array([4, 61, -3])


P, Q, L, U = lu_decomposition.lu_complete_pivot(rand_matrix2)

LU = np.dot(np.dot(P, np.dot(L, U)), Q)

P2, L2, U2 = sp.lu(rand_matrix2)

LU2 = np.dot(P2, np.dot(L2, U2))
