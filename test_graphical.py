import plotly.plotly as py
import plotly.graph_objs as go
import time
import numpy as np
import cholesky
import tests
import lu_arbitrary2
import scipy.linalg as sp
import lu_square
import tests
import solve
py.sign_in('kamp9', '09g4enb2lz')


# se int og real matrix
def precision_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test_sp = []
        test_my = []
        test_lu = []
        test_partial = []
        test_complete = []
        test_rook = []
        test_block = []
        for j in range(repeat):
            A = np.random.rand(i, i)
            Ab = np.random.rand(i, 1)
            #pos_def = tests.generate_pos_dif(1000, 1, 1000)
            #A = np.random.randint(-100000, 100000, size=(i, i))  #change
            #Ab = np.random.randint(-100000, 100000, size=(i, 1))        #change
            """
            x = sp.solve(A, Ab)
            Asp = np.dot(A, x)
            dif_matrix = Asp - Ab
            test_sp += [np.sum(np.abs(dif_matrix))/np.sum(Ab)]

            x = solve.solve(A, Ab, 4)
            Asp = np.dot(A, x)
            dif_matrix = Asp - Ab
            test_block += [np.sum(np.abs(dif_matrix))/np.sum(Ab)]

            invers = sp.inv(A)
            Asp = np.dot(A, invers)
            dif_matrix = Asp - np.identity(Asp.shape[0])
            test_sp += [np.sum(np.abs(dif_matrix))]

            invers = solve.inverse(A, 2)
            Asp = np.dot(A, invers)
            dif_matrix = Asp - np.identity(Asp.shape[0])
            test_my += [np.sum(np.abs(dif_matrix))]
            """

            """
            U2 = sp.cholesky(pos_def)
            Amy = np.dot(U2.transpose(), U2)
            dif_matrix = Amy - pos_def
            print np.sum(np.abs(dif_matrix))

            U2 = cholesky.cholesky_block(pos_def, 92)
            Amy = np.dot(U2.transpose(), U2)
            dif_matrix = Amy - pos_def
            print np.sum(np.abs(dif_matrix))

            U2 = cholesky.cholesky_out_of_place(pos_def)
            Amy = np.dot(U2.transpose(), U2)
            dif_matrix = Amy - pos_def
            print np.sum(np.abs(dif_matrix))


            L2, U2 = lu_square.lu_in_place(A)
            lu = np.dot(L2, U2)
            dif_matrix = lu - A
            test_lu += [np.sum(np.abs(dif_matrix))]

            print i, j
            P3, L3, U3 = lu_square.lu_partial_pivot(A)
            partial = np.dot(P3, np.dot(L3, U3))
            dif_matrix = partial - A
            test_partial += [np.sum(np.abs(dif_matrix))]

            P4, Q4, L4, U4 = lu_square.lu_complete_pivot(A)
            complete = np.dot(np.dot(P4, np.dot(L4, U4)), Q4)
            dif_matrix = complete - A
            test_complete += [np.sum(np.abs(dif_matrix))]

            P5, Q5, L5, U5 = lu_square.lu_rook_pivot(A)
            rook = np.dot(np.dot(P5, np.dot(L5, U5)), Q5)
            dif_matrix = rook - A
            test_rook += [np.sum(np.abs(dif_matrix))]

            P7, L7, U7 = lu_arbitrary2.lu_partial_block2(A, 42)
            block = np.dot(P7, np.dot(L7, U7))
            dif_matrix = block - A
            test_block += [np.sum(np.abs(dif_matrix))]
        """
            print i, j

        plot_data += [{
            'y': test_sp,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

#        plot_data += [{
#            'y': test_partial,
#            'type':'box',
#            'marker':{'color': 'blue'},
#            'name': str(i) + 'x' + str(i),
#            'boxpoints': False
#            }]

        plot_data += [{
            'y': test_my,
            'type':'box',
            'marker':{'color': 'blue'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
        """
        plot_data += [{
            'y': test_rook,
            'type':'box',
            'marker':{'color': 'green'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
        plot_data += [{
            'y': test_block,
            'type':'box',
            'marker':{'color': 'yellow'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
        plot_data += [{
            'y': test_block,
            'type':'box',
            'marker':{'color': 'yellow'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
        """
    url = py.plot(plot_data, filename='precision')

precision_test(100, 501, 100, 10)


def benchmark_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test_sp = []
        test_my = []
        for j in range(repeat):
            rand_int_matrix = tests.generate_pos_dif(i, 1000, 100000)

            time_start = time.clock()
            cholesky.cholesky_block(rand_int_matrix, 92)
            test_my += [time.clock() - time_start]

            time_start = time.clock()
            sp.cholesky(rand_int_matrix)
            test_sp += [time.clock() - time_start]

            print i, j
            plot_data += [{
                'y': test_my,
                'type':'box',
                'marker':{'color': 'red'},
                'name': str(i) + 'x' + str(i),
                'boxpoints': False
                }]
            plot_data += [{
                'y': test_sp,
                'type':'box',
                'marker':{'color': 'blue'},
                'name': str(i) + 'x' + str(i),
                'boxpoints': False
                }]

    url = py.plot(plot_data, filename='Benchmark')

#benchmark_test(100, 1001, 100, 10)


def block_test(minsize, maxsize, step, repeat):
    plot_data = []
    rand_matrix = np.random.rand(500, 500)  # change
    #pos_def = tests.generate_pos_dif(1000, 1000, 1000000)
    rand_int_matrix = np.random.randint(-1000, 1000, size=(1000, 1000))
    for i in xrange(minsize, maxsize, step):
        test = []
        for j in xrange(repeat):
            t0 = time.clock()
            lu_arbitrary2.lu_block(rand_matrix, i)
            test += [time.clock() - t0]
            print i, j
        plot_data += [{
            'y': test,
            'type':'box',
            'marker':{'color': 'black'},
            'name': ' ' + str(i),
            'boxpoints': False
            }]
    url = py.plot(plot_data, filename='Benchmark')


#block_test(1, 101, 1, 5)
