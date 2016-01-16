import scipy.linalg as sp
import numpy as np
import lu_block_slow
import lu_block
import lu_
import cholesky

A = np.random.rand(1500, 1500)

P, L, U = sp.lu(A)
Asp = np.dot(P, np.dot(L, U))

L2, U2 = lu_.lu_in_place(A)
Alu = np.dot(L2, U2)

P3, L3, U3 = lu_.lu_partial_pivot(A)
Alu_partial = np.dot(P3, np.dot(L3, U3))

P4, Q4, L4, U4 = lu_.lu_complete_pivot(A)
Alu_complete = np.dot(np.dot(P4, np.dot(L4, U4)), Q4)

P5, Q5, L5, U5 = lu_.lu_rook_pivot(A)
Alu_rook = np.dot(np.dot(P5, np.dot(L5, U5)), Q5)

L6, U6 = lu_.lu_block(A, 42)
Alu_block = np.dot(L6, U6)

P7, L7, U7 = lu_block.lu_partial_block(A, 42)
Alu_partial_block = np.dot(P7, np.dot(L7, U7))


def dif_check(new_A, original_A):
    dif_matrix = new_A - original_A
    return np.sum(np.abs(dif_matrix))

print 'sp'
print dif_check(Asp, A)
print 'lu'
print dif_check(Alu, A)
print 'lu_partial'
print dif_check(Alu_partial, A)
print 'lu_complete'
print dif_check(Alu_complete, A)
print 'lu_rook'
print dif_check(Alu_rook, A)
print 'lu_block'
print dif_check(Alu_block, A)
print 'lu_partial_block'
print dif_check(Alu_partial_block, A)
