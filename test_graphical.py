import plotly.plotly as py
import plotly.graph_objs as go
import time
import numpy as np
import cholesky
import tests
import lu_block
import scipy.linalg as sp
import lu_square
import tests
import solve_and_invese
py.sign_in('kamp9', '09g4enb2lz')


def norm_1(A, newA):
    dif_sq = np.abs(A - newA)
    return np.sum(dif_sq)


def norm_2(A, newA):
    dif_sq = np.square(A - newA)
    return np.sqrt(np.sum(dif_sq))/np.sqrt(np.sum(np.square(A)))



def precision_test(minsize, maxsize, step, repeat):
    plot_data  = []
    y_sp       = []
    y_partial  = []
    y_complete = []
    y_rook     = []
    y_block    = []
    for index, i in enumerate(range(minsize, maxsize, step)):
        test_sp       = []
        test_my       = []
        test_lu       = []
        test_partial  = []
        test_complete = []
        test_rook     = []
        test_block    = []
        for j in range(repeat):
            # A = np.random.rand(i, i)
            # Ab = np.random.rand(i, 1)
            # pos_def = tests.generate_pos_dif(1000, 1, 1000)
            A = np.random.randint(-1000, 1000, size=(i, i))   # change
            Ab = np.random.randint(-1000, 1000, size=(i, 1))  # change

            # x = sp.solve(A, Ab)
            # Anew = np.dot(A, x)
            # test_sp += [norm_2(Ab, Anew)]
            #
            # x = solve.solve(A, Ab, 2)
            # Anew = np.dot(A, x)
            # test_block += [norm_2(Ab, Anew)]

            # invers = sp.inv(A)
            # Asp = np.dot(A, invers)
            # dif_matrix = Asp - np.identity(Asp.shape[0])
            # test_sp += [np.sum(np.abs(dif_matrix))]

            # invers = solve.inverse(A, 2)
            # Asp = np.dot(A, invers)
            # dif_matrix = Asp - np.identity(Asp.shape[0])
            # test_my += [np.sum(np.abs(dif_matrix))]

            x = sp.solve(A, Ab)
            Amy = np.dot(A, x)
            test_sp += [norm_2(Ab, Amy)]

            # x = solve.solve(A, Ab, 1)
            # Amy = np.dot(A, x)
            # test_partial += [norm_2(Ab, Amy)]

            # x = solve.solve(A, Ab, 2)
            # Amy = np.dot(A, x)
            # test_complete += [norm_2(Ab, Amy)]

            # x = solve.solve(A, Ab, 3)
            # Amy = np.dot(A, x)
            # test_rook += [norm_2(Ab, Amy)]
            #
            x = solve_and_invese.solve(A, Ab, 4)
            Amy = np.dot(A, x)
            test_block += [norm_2(Ab, Amy)]

            # U2 = cholesky.cholesky_out_of_place(pos_def)
            # Amy = np.dot(U2.transpose(), U2)
            # dif_matrix = Amy - pos_def
            # print np.sum(np.abs(dif_matrix))

            # L2, U2 = lu_square.lu_in_place(A)
            # lu = np.dot(L2, U2)
            # dif_matrix = lu - A
            # test_lu += [np.sum(np.abs(dif_matrix))]

            # P3, L3, U3 = lu_square.lu_partial_pivot(A)
            # partial = np.dot(P3, np.dot(L3, U3))
            # dif_matrix = partial - A
            # test_partial += [np.sum(np.abs(dif_matrix))]
            #
            # P4, Q4, L4, U4 = lu_square.lu_complete_pivot(A)
            # complete = np.dot(np.dot(P4, np.dot(L4, U4)), Q4)
            # dif_matrix = complete - A
            # test_complete += [np.sum(np.abs(dif_matrix))]

            # P5, Q5, L5, U5 = lu_square.lu_rook_pivot(A)
            # rook = np.dot(np.dot(P5, np.dot(L5, U5)), Q5)
            # dif_matrix = rook - A
            # test_rook += [np.sum(np.abs(dif_matrix))]
            #

            print i, j

        plot_data += [{
            'y': test_sp,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        # plot_data += [{
        #     'y': test_partial,
        #     'type':'box',
        #     'marker':{'color': 'blue'},
        #     'name': str(i) + 'x' + str(i),
        #     'boxpoints': False
        #     }]
        #
        # plot_data += [{
        #     'y': test_complete,
        #     'type':'box',
        #     'marker':{'color': 'red'},
        #     'name': str(i) + 'x' + str(i),
        #     'boxpoints': False
        #     }]

        # plot_data += [{
        #     'y': test_rook,
        #     'type':'box',
        #     'marker':{'color': 'green'},
        #     'name': str(i) + 'x' + str(i),
        #     'boxpoints': False
        #     }]
        #
        plot_data += [{
            'y': test_block,
            'type':'box',
            'marker':{'color': 'purple'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        # sp_avg = [np.sum(test_sp) / len(test_sp)]
        # y += [sp_avg][0]
        # plot_data += [{
        #      'y': sp_avg,
        #      'type':'box',
        #      'marker':{'color': 'green'},
        #      'name': str(i) + 'x' + str(i)
        #      }]

        # print plot_data
        # xi = np.arange(len(avg))

        y_sp += [np.sum(test_sp) / len(test_sp)]
        y_partial += [np.sum(test_partial) / len(test_partial)]
        y_complete += [np.sum(test_complete) / len(test_complete)]
        y_rook += [np.sum(test_rook) / len(test_rook)]
        y_block += [np.sum(test_block) / len(test_block)]

    print np.polyfit(range(len(y_sp)), y_sp, 1, full=True)
    print np.polyfit(range(len(y_sp)), y_sp, 2, full=True)
    # print np.polyfit(range(len(y_partial)), y_partial, 1, full=True)
    # print np.polyfit(range(len(y_partial)), y_partial, 2, full=True)
    # print np.polyfit(range(len(y_complete)), y_complete, 1, full=True)
    # print np.polyfit(range(len(y_complete)), y_complete, 2, full=True)
    # print np.polyfit(range(len(y_rook)), y_rook, 1, full=True)
    # print np.polyfit(range(len(y_rook)), y_rook, 2, full=True)
    print np.polyfit(range(len(y_block)), y_block, 1, full=True)
    print np.polyfit(range(len(y_block)), y_block, 2, full=True)

    url = py.plot(plot_data, filename='precision')

#precision_test(100, 1001, 100, 10)


def benchmark_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test_sp = []
        test_my = []
        for j in range(repeat):
            rand_int_matrix = tests.generate_pos_dif(i, 1000, 100000)

            time_start = time.clock()
            lu_square.lu_out_of_place(rand_int_matrix)
            test_my += [time.clock() - time_start]

            time_start = time.clock()
            lu_square.lu_in_place(rand_int_matrix)
            test_sp += [time.clock() - time_start]

            print i, j
            plot_data += [{
                'y': test_my,
                'type':'box',
                'marker':{'color': 'orange'},
                'name': str(i) + 'x' + str(i),
                'boxpoints': False
                }]
            plot_data += [{
                'y': test_sp,
                'type':'box',
                'marker':{'color': 'cyan'},
                'name': str(i) + 'x' + str(i),
                'boxpoints': False
                }]

    url = py.plot(plot_data, filename='Benchmark')

benchmark_test(2001, 2002, 100, 4)


def block_test(minsize, maxsize, step, repeat):
    plot_data = []
    #rand_matrix = np.random.rand(500, 500)  # change
    #pos_def = tests.generate_pos_dif(1000, 1000, 1000000)
    rand_int_matrix = np.random.randint(-1000, 1000, size=(2000, 2000))
    for i in xrange(minsize, maxsize, step):
        test = []
        for j in xrange(repeat):
            t0 = time.clock()
            lu_block.lu_block(rand_int_matrix, i)
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


#block_test(4, 501, 4, 4)



"""
sp, partial, complete, rook, block. 2 for hver
(array([  6.47301487e-16,   1.40748478e-15]), array([  1.14604420e-31]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([ -1.42738132e-17,   7.75765806e-16,   1.23619902e-15]), array([  7.02877845e-33]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
(array([  6.47301487e-16,   1.40748478e-15]), array([  1.14604420e-31]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([ -1.42738132e-17,   7.75765806e-16,   1.23619902e-15]), array([  7.02877845e-33]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
(array([  4.31868055e-16,   1.02927773e-15]), array([  6.07129861e-32]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([ -1.04402565e-17,   5.25830364e-16,   9.03994656e-16]), array([  3.16153704e-33]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
(array([  5.03129492e-16,   1.15159889e-15]), array([  7.82705816e-32]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([ -1.17642057e-17,   6.09007343e-16,   1.01042843e-15]), array([  5.19721131e-33]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
(array([  1.20216020e-15,   1.08807213e-15]), array([  1.22426682e-31]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([ -1.40630267e-17,   1.32872744e-15,   9.19315806e-16]), array([  1.80047981e-32]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
"""

"""
lu uden pivot
(array([  4.31107011e-13,  -2.60381436e-13]), array([  8.28344472e-25]), 2, array([ 1.35754456,  0.39632407]), 2.2204460492503131e-15)
(array([  2.71976352e-14,   1.86328294e-13,   6.59901862e-14]), array([  4.37776874e-25]), 3, array([ 1.64219501,  0.53895301,  0.11280603]), 2.2204460492503131e-15)
"""

