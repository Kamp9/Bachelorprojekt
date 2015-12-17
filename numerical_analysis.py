import scipy.linalg as sp
import numpy as np
import lu_arbitrary


rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
P, L, U = lu_arbitrary.lu_partial_block2(rand_int_matrix, 68)
Alal = np.dot(P, np.dot(L, U))
print rand_int_matrix
print Alal

