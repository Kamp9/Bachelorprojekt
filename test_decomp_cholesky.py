from __future__ import division, print_function, absolute_import

from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
import cholesky_decomposition


class TestCholesky(TestCase):

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]
        c = cholesky_decomposition.cholesky(a)
        assert_array_almost_equal(np.dot(c, np.transpose(c)), a)

"""

    def test_check_finite(self):
        a = [[8,2,3],[2,9,3],[3,3,6]]
        c = cholesky(a, check_finite=False)
        assert_array_almost_equal(dot(transpose(c),c),a)
        c = transpose(c)
        a = dot(c,transpose(c))
        assert_array_almost_equal(cholesky(a,lower=1, check_finite=False),c)

    def test_simple_complex(self):
        m = array([[3+1j,3+4j,5],[0,2+2j,2+7j],[0,0,7+4j]])
        a = dot(transpose(conjugate(m)),m)
        c = cholesky(a)
        a1 = dot(transpose(conjugate(c)),c)
        assert_array_almost_equal(a,a1)
        c = transpose(c)
        a = dot(c,transpose(conjugate(c)))
        assert_array_almost_equal(cholesky(a,lower=1),c)

    def test_random(self):
        n = 20
        for k in range(2):
            m = random([n,n])
            for i in range(n):
                m[i,i] = 20*(.1+m[i,i])
            a = dot(transpose(m),m)
            c = cholesky(a)
            a1 = dot(transpose(c),c)
            assert_array_almost_equal(a,a1)
            c = transpose(c)
            a = dot(c,transpose(c))
            assert_array_almost_equal(cholesky(a,lower=1),c)

    def test_random_complex(self):
        n = 20
        for k in range(2):
            m = random([n,n])+1j*random([n,n])
            for i in range(n):
                m[i,i] = 20*(.1+abs(m[i,i]))
            a = dot(transpose(conjugate(m)),m)
            c = cholesky(a)
            a1 = dot(transpose(conjugate(c)),c)
            assert_array_almost_equal(a,a1)
            c = transpose(c)
            a = dot(c,transpose(conjugate(c)))
            assert_array_almost_equal(cholesky(a,lower=1),c)

"""
