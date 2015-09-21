# coding=utf-8
import numpy as np


def forward_substitution(L, b):
    (m, n) = L.shape
    z = np.zeros(m)
    for i in range(m):
        z[i] = (1.0 / L[i, i]) * (b[i] - np.dot(L[i, :i], z[:i]))
    z = np.resize(z, (m, 1))  # kan gøres pænere
    return z


def backward_substitution(U, z):
    (m, n) = U.shape
    n = m - 1                 # kan gøres pænere
    x = np.zeros(m)
    for i in range(m):
        x[n-i] = (1.0 / U[n-i, n-i]) * (z[n-i] - np.dot(U[n-i, n-i:], x[n-i:]))
    x = np.resize(x, (m, 1))  # kan gøres pænere
    return x

#  virker det stadig?