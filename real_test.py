from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
import scipy.linalg as sp
import cholesky_decomposition
import lu_decomposition
import solve


class TestCholesky(TestCase):

    def test_simple(self):
        rand_matrix2 = np.random.rand(500, 500)
        rand_col = np.random.rand(500, 1)
        # P, Q, L, U = lu_decomposition.lu_rook_pivot(rand_matrix2)
        # P2, L2, U2 = sp.lu(rand_matrix2)
        assert_array_almost_equal(sp.solve(rand_matrix2, rand_col), solve.solve(rand_matrix2, rand_col, 0), decimal=9)
