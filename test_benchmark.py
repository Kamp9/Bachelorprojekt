import cProfile
import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import lu_decomposition2
import solve
import lu_arbitrary
import lu_arbitrary2
import time

rand_matrix = np.random.rand(3000, 3000)

rand_col = np.random.rand(1000, 1)

a = np.random.random_integers(-1000, 1000, size=(2000, 2000))
b = np.random.random_integers(1000000, 100000000, size=(2000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)


# cProfile.run('cholesky.cholesky_block(a_sym, 100)')
# cProfile.run('sp.cholesky(a_sym)')
# cProfile.run('sp.lu(rand_matrix)')
# cProfile.run('lu.lu_block(rand_matrix, 42)')
# cProfile.run('lu.lu_inplace(rand_matrix)')
# cProfile.run('sp.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky2(a_sym)')

# cProfile.run('lu_arbitrary.lu_block(rand_matrix, 42)')

# cProfile.run('sp.lu(rand_matrix)')
# cProfile.run('lu_arbitrary2.lu_partial_block2(rand_matrix, 68)')

# cProfile.run('lu.lu_inplace(rand_matrix)')
# cProfile.run('lu.lu_inplace_with_dot(rand_matrix)')

# cProfile.run('rand_matrix.argmax()')
# cProfile.run('python_for(rand_matrix)')


# cProfile.run('cholesky.cholesky(a_sym)')
# cProfile.run('cholesky.cholesky2(a_sym)')

cProfile.run('sp.lu(rand_matrix)')
cProfile.run('lu_arbitrary2.lu_partial_block2(rand_matrix, 68)')


def lal():
    rand_matrix[[3, 5]] = rand_matrix[[5, 3]]


def lal2():
    temp = np.empty(200000)
    temp[:] = rand_matrix[3, :]
    rand_matrix[3, :] = rand_matrix[5, :]
    rand_matrix[5, :] = temp[:]

# cProfile.run('lal2()')
# cProfile.run('lal()')


def find_best_blocksize():
    t0 = time.clock()
    lu_arbitrary.lu_partial_block2(rand_matrix, 1)
    best_time = time.clock() - t0
    best_block = 1
    print best_time
    print
    for i in range(2, 1001):
        t0 = time.clock()
        lu_arbitrary.lu_partial_block2(rand_matrix, i)
        new_time = time.clock() - t0
        if new_time < best_time:
            best_time = new_time
            best_block = i
        print 'i : ' + str(i)
        print 'i time: ' + str(new_time)
        print 'best time: ' + str(best_time)
        print 'best blocksize : ' + str(best_block)
        print
#find_best_blocksize()

"""
116: 68 er bedst med 12.787971 for 2000 x 2000
"""

