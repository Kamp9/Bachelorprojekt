import cProfile
import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import lu_decomposition2
import solve
import lu_block_slow
import lu_block
import time
np.set_printoptions(linewidth=200)

rand_matrix = np.random.rand(2000, 2000)

rand_col = np.random.rand(1000, 1)

a = np.random.random_integers(-1000, 1000, size=(2000, 2000))
b = np.random.random_integers(1000000, 100000000, size=(2000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)


cProfile.run('cholesky.cholesky_block(a_sym, 100)')
cProfile.run('sp.cholesky(a_sym)')
cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu.lu_block(rand_matrix, 42)')
cProfile.run('lu.lu_inplace(rand_matrix)')
cProfile.run('sp.cholesky(a_sym)')
cProfile.run('cholesky.cholesky(a_sym)')
cProfile.run('cholesky.cholesky2(a_sym)')

cProfile.run('lu_arbitrary.lu_block(rand_matrix, 42)')

cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu_arbitrary2.lu_partial_block2(rand_matrix, 68)')

cProfile.run('lu.lu_inplace(rand_matrix)')
cProfile.run('lu.lu_inplace_with_dot(rand_matrix)')

cProfile.run('rand_matrix.argmax()')
cProfile.run('python_for(rand_matrix)')


cProfile.run('cholesky.cholesky(a_sym)')
cProfile.run('cholesky.cholesky2(a_sym)')

cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu_arbitrary2.lu_partial_block2(rand_matrix, 132)')


def advanced():
    rand_matrix[[3, 5]] = rand_matrix[[5, 3]]


def basic():
    temp = np.empty(200000)
    temp[:] = rand_matrix[3, :]
    rand_matrix[3, :] = rand_matrix[5, :]
    rand_matrix[5, :] = temp[:]

