import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
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

posdef_matrix = np.array([[2, -1, 0],
                          [-1, 2, -1],
                          [0, -1, 2]])

b3 = np.array([41, 31, 64])

a = np.array([[5315353, -5313451, 65153511], [-1355351, -35315, 35362], [0.00005353135, -5, 13646]])
b4 = np.array([4133, 6421, -533, -5533])


P, Q, L, U = lu_decomposition.lu_rook_pivot(rooktest)

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

lal = np.array(([[100]]))

lala_invers = np.array(([-1, 0, 0],
                        [-42, 2, 0],
                        [1, -31, -13]))

lal2 = np.array(([100, 42],
                 [42, 143]))

print solve.inverse(rooktest)
print np.dot(solve.inverse(rooktest), rooktest)
print np.dot(sp.inv(rooktest), rooktest)

