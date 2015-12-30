import numpy as np

@profile
def cholesky(A):
    A = A.astype(np.float64)
    m, n = A.shape
    U = np.zeros((m, m))
    for k in range(m):
        pivot = np.math.sqrt(A[k, k])
        U[k, k] = pivot
        U[k, k+1:] = A[k, k+1:] / pivot
        A[k+1:, k+1:] -= U[k, k+1:] * U[k, k+1:, np.newaxis]
    return U


a = np.random.random_integers(-1000, 1000, size=(1000, 1000))
b = np.random.random_integers(1000000, 100000000, size=(1000, 1))
a_sym = (a + a.T)/2
np.fill_diagonal(a_sym, b)

cholesky(a_sym)

