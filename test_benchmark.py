import cProfile
import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import lu_decomposition2
import solve

rand_matrix = np.random.rand(1000, 1000)
rand_col = np.random.rand(1000, 1)


a = np.random.random_integers(-1000, 1000, size=(2000, 2000))
b = np.random.random_integers(1000000, 100000000, size=(2000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)


# cProfile.run('cholesky.cholesky_block(a_sym, 100)')
# cProfile.run('sp.cholesky(a_sym)')


# cProfile.run('sp.solve(rand_matrix, rand_col)')
# cProfile.run('solve.solve(rand_matrix, rand_col, 3)')

cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu.lu_inplace(rand_matrix)')
cProfile.run('lu.lu_out_of_place(rand_matrix)')



# cProfile.run('sp.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky2(a_sym)')
