import plotly.plotly as py
import plotly.graph_objs as go
import time
import numpy as np
import cholesky
import tests
import lu_block
import scipy.linalg as sp
import lu
import tests
import solve
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
            # pos_def = tests.generate_pos_dif(i, 1, 1000)
            A = np.random.randint(-1000, 1000, size=(i, i))   # change
           # Ab = np.random.randint(-1000, 1000, size=(i, 1))  # change

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

            # x = sp.inv(A)
            # Amy = np.dot(A, x)
            # test_sp += [norm_2(np.identity(i), Amy)]
            #
            # # x = lu.lu_partial_pivot(A, Ab, 1)
            # # Amy = np.dot(A, x)
            # # test_partial += [norm_2(Ab, Amy)]
            #
            # x = solve.inverse(A, 4)
            # Amy = np.dot(A, x)
            # test_block += [norm_2(np.identity(i), Amy)]

            #
            # x = solve.solve(A, Ab, 3)
            # Amy = np.dot(A, x)
            # test_rook += [norm_2(Ab, Amy)]
            #
            # x = solve.solve(A, Ab, 4)
            # Amy = np.dot(A, x)
            # test_block += [norm_2(Ab, Amy)]

            P, L, U = sp.lu(A)
            Amy = np.dot(P, np.dot(L, U))
            test_sp += [norm_1(A, Amy)]

            P, L, U = lu.lu_partial_pivot(A)
            Amy = np.dot(P, np.dot(L, U))
            test_partial += [norm_1(A, Amy)]

            P, Q, L, U = lu.lu_complete_pivot(A)
            Amy = np.dot(np.dot(P, np.dot(L, U)), Q)
            test_complete += [norm_1(A, Amy)]

            P, Q, L, U = lu.lu_rook_pivot(A)
            Amy = np.dot(np.dot(P, np.dot(L, U)), Q)
            test_rook += [norm_1(A, Amy)]

            P, L, U = lu_block.lu_partial_block(A, 32)
            Amy = np.dot(P, np.dot(L, U))
            test_block += [norm_1(A, Amy)]

            # U = sp.cholesky(pos_def)
            # Amy = np.dot(U.transpose(), U)
            # test_sp += [norm_2(pos_def, Amy)]
            #
            # U = cholesky.cholesky_block(pos_def, 32)
            # Amy = np.dot(U.transpose(), U)
            # test_partial += [norm_2(pos_def, Amy)]

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

        plot_data += [{
            'y': test_partial,
            'type':'box',
            'marker':{'color': 'blue'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        plot_data += [{
            'y': test_complete,
            'type':'box',
            'marker':{'color': 'red'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

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
    print np.polyfit(range(len(y_partial)), y_partial, 1, full=True)
    print np.polyfit(range(len(y_partial)), y_partial, 2, full=True)
    print np.polyfit(range(len(y_complete)), y_complete, 1, full=True)
    print np.polyfit(range(len(y_complete)), y_complete, 2, full=True)
    print np.polyfit(range(len(y_rook)), y_rook, 1, full=True)
    print np.polyfit(range(len(y_rook)), y_rook, 2, full=True)
    print np.polyfit(range(len(y_block)), y_block, 1, full=True)
    print np.polyfit(range(len(y_block)), y_block, 2, full=True)

    url = py.plot(plot_data, filename='precision')

#precision_test(200, 2001, 200, 4)


def benchmark_test(minsize, maxsize, step, repeat):
    plot_data = []
    y_sp       = []
    y_block  = []
    y_no = []
    for i in range(minsize, maxsize, step):
        test_sp = []
        test_no = []
        test_partial = []
        test_complete = []
        test_rook = []
        test_block = []

        for j in range(repeat):
            # pos_def = tests.generate_pos_dif(i, 1, 1000)
            A = np.random.randint(-1000, 1000, size=(i, i))   # change
           # Ab = np.random.randint(-1000, 1000, size=(i, 1))  # change

            time_start = time.clock()
            sp.lu(A)
            test_sp += [time.clock() - time_start]

            time_start = time.clock()
            lu.lu_in_place(A)
            test_no += [time.clock() - time_start]

            time_start = time.clock()
            lu.lu_partial_pivot(A)
            test_partial += [time.clock() - time_start]

            time_start = time.clock()
            lu.lu_complete_pivot(A)
            test_complete += [time.clock() - time_start]

            time_start = time.clock()
            lu.lu_rook_pivot(A)
            test_rook += [time.clock() - time_start]

            time_start = time.clock()
            lu_block.lu_partial_block(A, 32)
            test_block += [time.clock() - time_start]


            # time_start = time.clock()
            # lu.lu_partial_pivot(A)
            # test_no += [time.clock() - time_start]

            print i, j

        plot_data += [{
            'y': test_sp,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        plot_data += [{
            'y': test_no,
            'type':'box',
            'marker':{'color': 'grey'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        plot_data += [{
            'y': test_partial,
            'type':'box',
            'marker':{'color': 'blue'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

        plot_data += [{
            'y': test_complete,
            'type':'box',
            'marker':{'color': 'red'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]

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
            'marker':{'color': 'purple'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
            # plot_data += [{
            #     'y': test_no,
            #     'type':'box',
            #     'marker':{'color': 'blue'},
            #     'name': str(i) + 'x' + str(i),
            #     'boxpoints': False
            # #     }]
            #
            # y_sp += [np.sum(test_sp) / len(test_sp)]
            # y_block += [np.sum(test_block) / len(test_block)]
            # # y_no += [np.sum(test_no) / len(test_no)]

    # print np.polyfit(range(len(y_sp)), y_sp, 1, full=True)
    # print np.polyfit(range(len(y_sp)), y_sp, 2, full=True)
    # print np.polyfit(range(len(y_block)), y_block, 1, full=True)
    # print np.polyfit(range(len(y_block)), y_block, 2, full=True)
    # # # print np.polyfit(range(len(y_no)), y_no, 1, full=True)
    # # # print np.polyfit(range(len(y_no)), y_no, 2, full=True)


    url = py.plot(plot_data, filename='Benchmark')

benchmark_test(200, 2001, 200, 4)


def block_test(minsize, maxsize, step, repeat):
    plot_data = []
    #rand_matrix = np.random.rand(500, 500)  # change
    #pos_def = tests.generate_pos_dif(1000, 1000, 1000000)
    rand_int_matrix = np.random.randint(-1000, 1000, size=(1500, 1500))
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


#block_test(10, 70, 1, 4)



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

"""
SP vs Block vs ikke block
(array([ 0.27621989, -2.17783483]), array([ 71.47326403]), 2, array([ 1.36401128,  0.37346114]), 8.8817841970012523e-15)
(array([ 0.01044912, -0.13129575,  0.40309755]), array([ 9.55365804]), 3, array([ 1.64814728,  0.52277869,  0.10155286]), 8.8817841970012523e-15)
(array([ 0.37224716, -2.72476711]), array([ 112.71332054]), 2, array([ 1.36401128,  0.37346114]), 8.8817841970012523e-15)
(array([ 0.01309176, -0.13833148,  0.50889764]), array([ 15.51362282]), 3, array([ 1.64814728,  0.52277869,  0.10155286]), 8.8817841970012523e-15)
(array([ 0.79877836, -6.08374302]), array([ 534.15627288]), 2, array([ 1.36401128,  0.37346114]), 8.8817841970012523e-15)
(array([ 0.02888144, -0.3275979 ,  1.04997333]), array([ 61.1067546]), 3, array([ 1.64814728,  0.52277869,  0.10155286]), 8.8817841970012523e-15)
"""

"""
hastighed for SP solve vs solve med blok
(array([ 0.26333138, -2.08036845]), array([ 63.81775624]), 2, array([ 1.36401128,  0.37346114]), 8.8817841970012523e-15)
(array([ 0.00992129, -0.12359881,  0.37018938]), array([ 7.99582676]), 3, array([ 1.64814728,  0.52277869,  0.10155286]), 8.8817841970012523e-15)
(array([ 0.36932482, -2.66518351]), array([ 102.26626305]), 2, array([ 1.36401128,  0.37346114]), 8.8817841970012523e-15)
(array([ 0.01260493, -0.12226743,  0.44823409]), array([ 12.16109905]), 3, array([ 1.64814728,  0.52277869,  0.10155286]), 8.8817841970012523e-15)
"""

