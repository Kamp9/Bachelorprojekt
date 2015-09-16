import numpy as np
import cholesky_decomposition
import lu_decomposition
import scipy
import scipy.linalg


real_matrix = np.array([[3.3821, 0.8784, 0.3613, -2.0349],
                        [0.8784, 2.0068, 0.5587, 0.1169],
                        [0.3613, 0.5587, 3.6656, 0.7807],
                        [-2.0349, 0.1169, 0.7807, 2.5397]])

real_matrix2 = np.array([[1, 3, 5],
                         [2, 4, 7],
                         [1, 1, 0]])

imag_matrix = np.array([[1.+0.j, 0.-2.j],
                        [0.+2.j, 5.+0.j]])

rand_matrix = np.random.random((2, 2))

matrix1 = np.array([[0.5],
                    [0.75]])

matrix2 = np.array([2, 9])

LU = matrix1 * matrix2

A = np.array([[9, 4],
              [7, 9]])

b = np.array([[7],
              [-4.153],
              [43],
              [61.13]])

(P, L, U) = scipy.linalg.lu(real_matrix2)

print L
print U
print P
print

print real_matrix2
print
print np.dot(L, U)
