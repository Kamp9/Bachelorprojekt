from guppy import hpy
import numpy as np
import cholesky


h = hpy()


def foo():
    a = np.random.random_integers(-1000, 1000, size=(1000, 1000))
    b = np.random.random_integers(1000000, 100000000, size=(1000, 1))
    a_sym = (a + a.T)/2
    np.fill_diagonal(a_sym, b)
    cholesky.cholesky(a_sym)

print h.foo()
