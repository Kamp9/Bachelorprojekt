import numpy as np
import scipy.linalg as sp
import lu
import cholesky
import lu_block
np.set_printoptions(linewidth=200)


def generate_pos_dif(n, fr, to):
    a = np.random.random_integers(fr, to, size=(n, n))
    b = np.random.random_integers(to * 10, to * 100, size=(n, 1))
    pos_dif = (a + a.T)/2
    np.fill_diagonal(pos_dif, b)
    return pos_dif
