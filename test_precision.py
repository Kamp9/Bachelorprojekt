import scipy.linalg as sp
import numpy as np
import lu_arbitrary
import lu_arbitrary2
import lu

rand_int_matrix = np.random.randint(-1000, 1000, size=(2000, 2000))
P, L, U = lu_arbitrary.lu_partial_block2(rand_int_matrix, 68)
Alal = np.dot(P, np.dot(L, U))
P2, L2, U2 = sp.lu(rand_int_matrix)
splal = np.dot(P2, np.dot(L2, U2))

P3, Q3, L3, U3 = lu.lu_complete_pivot(rand_int_matrix)

L4, U4 = lu.lu_in_place(rand_int_matrix)

inplace = np.dot(L4, U4)

completelal = np.dot(np.dot(P3, np.dot(L3, U3)), Q3)

P5, L5, U5 = lu_arbitrary2.lu_partial_block2(rand_int_matrix, 42)
lu_arbitrary2lal = np.dot(P5, np.dot(L5, U5))

P6, Q6, L6, U6 = lu.lu_rook_pivot(rand_int_matrix)
rooklal = np.dot(np.dot(P6, np.dot(L6, U6)), Q6)


def dif_check(new_A, original_A):
    sum = 0
    for i, ei in enumerate(new_A):
        for j, ej in enumerate(ei):
            sum += abs(ej - original_A[i, j])
    return sum


print dif_check(Alal, rand_int_matrix)
print dif_check(completelal, rand_int_matrix)
print dif_check(splal, rand_int_matrix)
print dif_check(inplace, rand_int_matrix)
print dif_check(lu_arbitrary2lal, rand_int_matrix)
print dif_check(rooklal, rand_int_matrix)
