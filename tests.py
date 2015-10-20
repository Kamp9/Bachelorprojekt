import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
import lu_decomposition2
import solve
from numpy.testing import TestCase, assert_array_almost_equal

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

rooktest = np.array([[155, 53113531513, 642, 531],
                     [-4, 553, 700, 1],
                     [-90053, 53536464, 513, 35],
                     [-53353, 53, 1000, -353]])

a = np.array([[5315353, -5313451, 65153511], [-1355351, -35315, 35362], [0.00005353135, -5, 13646]])
b = np.array([4133, 6421, -533, -5533])


P, Q, L, U = lu_decomposition.lu_rook_pivot(rooktest)

P2, L2, U2, Q2, n = lu_decomposition2.lu_decompose(rooktest, 0)


print P2
print Q2
print L2
print U2
print n

print P2, np.dot(L2, U2)

