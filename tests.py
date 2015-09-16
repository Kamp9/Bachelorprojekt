import numpy as np
import cholesky_decomposition
import lu_decomposition

real_matrix = np.array([[3.3821, 0.8784, 0.3613, -2.0349],
                        [0.8784, 2.0068, 0.5587, 0.1169],
                        [0.3613, 0.5587, 3.6656, 0.7807],
                        [-2.0349, 0.1169, 0.7807, 2.5397]])

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


x = cholesky_decomposition.solve(real_matrix, b)

print np.dot(real_matrix, x)
