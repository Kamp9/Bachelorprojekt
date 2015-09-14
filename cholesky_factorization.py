from numpy import *

"""
def is_pos_def2(A):
    (m, n) = A.shape
    if m != n:
        return False
    for i in range(m):
        subA = A[0:i, 0:i]
        if linalg.det(A) <= 0:
            return False
    return True
"""


def is_pos_def(A):
    return all(linalg.eigvals(A) > 0)


def cholesky(A):
    if is_pos_def(A):
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
    else:
        raise ValueError('Matrix is not positive definite')

