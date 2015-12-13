import cProfile
import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import lu_decomposition2
import solve
import lu_arbitrary

rand_matrix = np.random.rand(2500, 2500)

"""
rand_col = np.random.rand(1000, 1)


a = np.random.random_integers(-1000, 1000, size=(2000, 2000))
b = np.random.random_integers(1000000, 100000000, size=(2000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)
"""

# cProfile.run('cholesky.cholesky_block(a_sym, 100)')
# cProfile.run('sp.cholesky(a_sym)')
# cProfile.run('sp.lu(rand_matrix)')
# cProfile.run('lu.lu_block(rand_matrix, 42)')
# cProfile.run('lu.lu_inplace(rand_matrix)')
# cProfile.run('sp.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky2(a_sym)')

#cProfile.run('lu_arbitrary.lu_block(rand_matrix, 42)')
cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu_arbitrary.lu_partial_block2(rand_matrix, 125)')
cProfile.run('lu.lu_block(rand_matrix, 125)')
cProfile.run('lu.lu_partial_pivot(rand_matrix)')


