from numpy import *


def cholesky(A):
    (m, n) = A.shape
    L = zeros((m, m))
    for i in range(m):
        a00 = math.sqrt(A[0, 0])
        L[i, i] = a00
        if m != 1:
            L10 = A[1:, 0] / a00
            L[i+1:, i] = L10
            A = A[1:, 1:] - L10[:, newaxis] * L10[newaxis, :]
            (m, n) = A.shape
    return L

matrix = array([(3.3821, 0.8784, 0.3613, -2.0349),
                (0.8784, 2.0068, 0.5587, 0.1169),
                (0.3613, 0.5587, 3.6656, 0.7807),
                (-2.0349, 0.1169, 0.7807, 2.5397)])


