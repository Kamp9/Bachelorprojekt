from numpy import array, argmax, dot, identity, newaxis, abs, empty, unravel_index;


def find_pivot(U, k, pivoting):
    # partial pivoting
    if pivoting == 0:
        return argmax(abs(U[k+1:, k])) + k+1, k

    # complete pivoting
    if pivoting == 1:
        Ucorner = U[k+1:, k+1:]
        return array(unravel_index(argmax(abs(Ucorner)), Ucorner.shape)) + (k+1, k+1)

    # rook pivoting
    if pivoting == 2:
        rowindex = argmax(abs(U[k+1:, k]))+k+1
        colmax = abs(U[rowindex][k])
        rowmax = -1
        while rowmax < colmax:
            colindex = argmax(abs(U[rowindex, k+1:]))+ k+1
            rowmax = abs(U[rowindex][colindex])
            if colmax < rowmax:
                rowindex = argmax(abs(U[k+1:, colindex]))+ k+1
                colmax = abs(U[rowindex][colindex])
            else:
                break
        return rowindex, colindex


# Permutation til LU-faktoriseringsskridt
def permute_PLUQ((i, j), P, L, U, Q, k):
    # Permuter raekker
    if i != k:
        U[[i, k], k:] = U[[k, i], k:]
        L[[i, k], :k] = L[[k, i], :k]
        P[i], P[k] = P[k], P[i]

    # Permuter soejler
    if j != k:
        U[:, [j, k]] = U[:, [k, j]]
        Q[i], Q[k] = Q[k], Q[i]


# Find (P,L,U,Q) saa P,Q er permutationer (i -> P[i], j -> Q[j]), L nedre-triangulaer med 1-diagonal,
# og U er oevre-triangulaer. 
def lu_decompose(A, pivot_scheme=0):
    n = len(A)
    L = identity(n)
    U = A.astype(float)
    P, Q = range(n), range(n)

    for k in range(n-1):
        i, j = find_pivot(U, k, pivot_scheme)
        permute_PLUQ((i, j), P, L, U, Q, k)
        pivot = U[k, k]
        if pivot == 0:          # A has rank < n
            return P, L, U, Q, k

        L[k+1:, k] = (1/pivot) * U[k+1:, k]
        U[k+1:, k+1:] -= (1/pivot) * U[k+1:, k, newaxis] * U[k, k+1:]
        U[k+1:, k] = 0

    return P, L, U, Q, n


"""
# Givet permutation i -> P[i], permuter raekker
def permute_rows(P,A):
    PA = empty(A.shape);
    for i in range(len(P)):
        PA[P[i], :] = A[i,:];
    return PA;

# Givet permutation j -> Q[j], permuter soejler
def permute_cols(Q,A):
    AQ = empty(A.shape);
    for j in range(len(Q)):
        AQ[:,Q[j]] = A[:,j];
    return AQ;

"""