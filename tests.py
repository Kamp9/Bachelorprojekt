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


rand_matrix2 = np.random.rand(4, 4)
rand_col = np.random.rand(4, 1)

lalala = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rooktest = np.array([[155, -800, 642, 531],
                     [-4, 553, 700, 1],
                     [-900, 732, 513, 35],
                     [-3, 53, 1, -353]])

a = np.array([[5315353, -5313451, 65153511], [-1355351, -35315, 35362], [0.00005353135, -5, 13646]])
b = np.array([4133, 6421, -533, -5533])


P, Q, L, U = lu_decomposition.lu_complete_pivot(rooktest)

LU = np.dot(np.dot(P, np.dot(L, U)), Q)

P2, L2, U2 = sp.lu(rooktest)

x = solve.solve(rooktest, b, 1)
print x
print sp.solve(rand_matrix2, rand_col)
