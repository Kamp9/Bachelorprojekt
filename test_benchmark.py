import cProfile
import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
import lu_decomposition2
import solve

rand_matrix = np.random.rand(1000, 1000)
rand_col = np.random.rand(1000, 1)

# cProfile.run('sp.solve(rand_matrix, rand_col)')
cProfile.run('solve.solve(rand_matrix, rand_col, 3)')
