#import plotly.plotly as py
#import plotly.graph_objs as go
import time
import numpy as np
import cholesky
import lu
import tests
import scipy.linalg as sp

#py.sign_in('kamp9', '09g4enb2lz')


"""
P, L, U = sp.lu(A)
Asp = np.dot(P, np.dot(L, U))

L2, U2 = lu.lu_in_place(A)
Alu = np.dot(L2, U2)

P3, L3, U3 = lu.lu_partial_pivot(A)
Alu_partial = np.dot(P3, np.dot(L3, U3))

P4, Q4, L4, U4 = lu.lu_complete_pivot(A)
Alu_complete = np.dot(np.dot(P4, np.dot(L4, U4)), Q4)

P5, Q5, L5, U5 = lu.lu_rook_pivot(A)
Alu_rook = np.dot(np.dot(P5, np.dot(L5, U5)), Q5)

L6, U6 = lu.lu_block(A, 42)
Alu_block = np.dot(L6, U6)

P7, L7, U7 = lu_arbitrary2.lu_partial_block2(A, 42)
Alu_partial_block = np.dot(P7, np.dot(L7, U7))
"""

f = open('workfile', 'r+')


# se int og real matrix
def precision_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test = []
        for j in range(repeat):
            A = np.random.rand(i, i)        # change
            P, L, U = sp.lu(A)
            Asp = np.dot(P, np.dot(L, U))   # change
            dif_matrix = Asp - A
            test += [np.sum(np.abs(dif_matrix))]
        plot_data += [{
            'y': test,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
    f.write(plot_data)

#  precision_test(500, 3001, 500, 10)


def benchmark_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test = []
        for j in range(repeat):
            rand_int_matrix = np.random.randint(-100000, 100000, size=(i, i))
            time_start = time.clock()
            sp.lu(rand_int_matrix)  # change funktion
            test += [time.clock() - time_start]
        plot_data += [{
            'y': test,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
    f.write(plot_data)


benchmark_test(10, 201, 1, 10)

f.close()
