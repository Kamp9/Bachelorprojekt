from numpy import *
import cholesky_decomposition
import lu_decomposition

real_matrix = array([[3.3821, 0.8784, 0.3613, -2.0349],
                     [0.8784, 2.0068, 0.5587, 0.1169],
                     [0.3613, 0.5587, 3.6656, 0.7807],
                     [-2.0349, 0.1169, 0.7807, 2.5397]])

imag_matrix = array([[1.+0.j, 0.-2.j],
                     [0.+2.j, 5.+0.j]])

rand_matrix = random.random((2, 2))

print cholesky_decomposition.cholesky(real_matrix)

matrix1 = array([[0.5],
                 [0.75]])

matrix2 = array([2, 9])

LU = matrix1 * matrix2

A = array([[9, 4],
           [7, 9]])

print A - LU

