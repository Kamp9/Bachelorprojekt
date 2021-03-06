# coding=utf-8
import numpy as np


# Checks if a given matrix is positive-definite.
def is_pos_def(A):
    return all(np.linalg.eigvals(A) > 0)


# Returns the Cholesky decomposition U for A=UtU of a positive-definite matrix A.
# Using the two matrices A and U, where astype(np.float64) copies the matrix A
def cholesky_out_of_place(A):
    A = A.astype(np.float64)
    m, n = A.shape
    U = np.zeros((m, m))
    for k in range(m):
        U[k, k] = np.sqrt(A[k, k])
        U[k, k+1:] = A[k, k+1:] / U[k, k]
        A[k+1:, k+1:] -= U[k, k+1:, np.newaxis] * U[k, k+1:]
    return U


# Returns the Cholesky decomposition U for A=UtU of a positive-definite matrix A.
# Using only one matrix U, where astype(np.float64) copies the matrix A
def cholesky_in_place(A):
    U = A.astype(np.float64)
    m, n = A.shape
    for k in range(m):
        U[k, k] = np.math.sqrt(U[k, k])
        U[k, k+1:] = U[k, k+1:] / U[k, k]
        U[k+1:, k+1:] -= U[k, k+1:] * U[k, k+1:, np.newaxis]
    return np.triu(U)


def forward_block_substitution(U, B):
    m, n = U.shape
    r, n = B.shape
    x = np.zeros((r, n))
    for k in range(m):
        x[k] = (B[k] - np.dot(U[k, :k], x[:k])) / U[k, k]
    return x


# Returns the Cholesky decomposition U for A=UtU of a positive-definite matrix A.
# Using a block algorithem to minimize cache misses.
def cholesky_block(A, r):
    m, n = A.shape
    U = np.zeros((m, m))
    A = A.astype(np.float64)
    for k in range(0, m, r):
        U[k:k+r, k:k+r] = cholesky_out_of_place(A[k:k+r, k:k+r])
        U[k:k+r, k+r:] = forward_block_substitution(U[k:k+r, k:k+r].transpose(), A[k:k+r, k+r:])
        A[k+r:, k+r:] -= np.dot(U[k:k+r, k+r:].transpose(), U[k:k+r, k+r:])
    return U

