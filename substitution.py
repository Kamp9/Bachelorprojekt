# coding=utf-8
import numpy as np


def forward_substitution(L, b):
    (m, n) = L.shape
    z = np.zeros(m)
    for k in range(m):
        z[k] = (1.0 / L[k, k]) * (b[k] - np.dot(L[k, :k], z[:k]))
    z = z[:, np.newaxis]
    return z


def backward_substitution(U, z):
    (m, n) = U.shape
    l = m - 1
    x = np.zeros(m)
    for k in range(m):
        x[l-k] = (1.0 / U[l-k, l-k]) * (z[l-k] - np.dot(U[l-k, l-k:], x[l-k:]))
    x = x[:, np.newaxis]
    return x

