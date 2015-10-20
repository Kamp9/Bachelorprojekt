from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
import solve


class TestLinAlg(TestCase):
    def test_simple(self):
        rand_matrix = np.random.rand(1000, 1000)
        rand_col = np.random.rand(1000, 1)
        sp_solve = sp.solve(rand_matrix, rand_col)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 0), decimal=7)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 1), decimal=7)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 2), decimal=7)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 3), decimal=7)

        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        rand_int_col = np.random.randint(-1000, 1000, size=(1000, 1))
        sp_solve2 = sp.solve(rand_int_matrix, rand_int_col)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 0), decimal=7)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 1), decimal=7)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 2), decimal=7)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 3), decimal=7)

