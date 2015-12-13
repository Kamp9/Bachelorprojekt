from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
import scipy.linalg as sp
import cholesky
import lu
import solve
import unittest
import tests
import lu_arbitrary


class TestLinAlg(unittest.TestCase):
    def test_lu_arbitrary(self):
        a_sym = tests.generate_pos_dif(1000, -1000, 1000)
        assert_array_almost_equal(lu_arbitrary.lu_block(a_sym, 42)[0], sp.lu(a_sym)[1], decimal=12)
        assert_array_almost_equal(lu_arbitrary.lu_block(a_sym, 42)[1], sp.lu(a_sym)[2], decimal=8)

    def test_lu_block(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        assert_array_almost_equal(lu.lu_inplace(rand_int_matrix)[0], lu.lu_block(rand_int_matrix, 10)[0], decimal=2)

    def test_lu_block_arbitrary(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        assert_array_almost_equal(lu_arbitrary.lu_partial(rand_int_matrix)[1], sp.lu(rand_int_matrix)[1], decimal=12)
        assert_array_almost_equal(lu_arbitrary.lu_partial(rand_int_matrix)[2], sp.lu(rand_int_matrix)[2], decimal=8)

    def test_lu_block_arbitrary2(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        np.array_equal(lu_arbitrary.lu_partial_block2(rand_int_matrix, 22)[0], sp.lu(rand_int_matrix)[0])
        assert_array_almost_equal(lu_arbitrary.lu_partial_block2(rand_int_matrix, 22)[1], sp.lu(rand_int_matrix)[1], decimal=12)
        assert_array_almost_equal(lu_arbitrary.lu_partial_block2(rand_int_matrix, 22)[2], sp.lu(rand_int_matrix)[2], decimal=8)

"""
    def test_cholesky(self):
        a = np.random.random_integers(-1000, 1000, size=(1000, 1000))
        b = np.random.random_integers(1000000, 100000000, size=(1000, 1))
        a_sym = (a + a.T)/2
        np.fill_diagonal(a_sym, b)
        assert_array_almost_equal(sp.cholesky(a_sym), cholesky.cholesky(a_sym), decimal=10)

    def test_cholesky_block(self):
        a_sym = tests.generate_pos_dif(2000, -1000, 1000)
        assert_array_almost_equal(sp.cholesky(a_sym), cholesky.cholesky_block(a_sym, 100), decimal=10)

    def test_lu_for_floats(self):
        rand_matrix = np.random.rand(1000, 1000)
        rand_col = np.random.rand(1000, 1)
        sp_solve = sp.solve(rand_matrix, rand_col)

        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 0), decimal=6)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 1), decimal=6)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 2), decimal=6)
        assert_array_almost_equal(sp_solve, solve.solve(rand_matrix, rand_col, 3), decimal=6)

    def test_lu_for_ints(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        rand_int_col = np.random.randint(-1000, 1000, size=(1000, 1))
        sp_solve2 = sp.solve(rand_int_matrix, rand_int_col)

        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 1), decimal=9)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 2), decimal=9)
        assert_array_almost_equal(sp_solve2, solve.solve(rand_int_matrix, rand_int_col, 3), decimal=9)

    def test_inverse(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(100, 100))
        assert_array_almost_equal(sp.inv(rand_int_matrix), solve.inverse(rand_int_matrix), decimal=12)

    def test_lu_arbitrary_partial(self):
        rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
        assert_array_almost_equal(lu_arbitrary.lu_partial_pivot(rand_int_matrix, 42)[1], sp.lu(rand_int_matrix)[1], decimal=8)
        assert_array_almost_equal(lu_arbitrary.lu_partial_pivot(rand_int_matrix, 42)[2], sp.lu(rand_int_matrix)[2], decimal=8)
"""


if __name__ == '__main__':
    TestLinAlg()

