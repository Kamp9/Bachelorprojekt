import plotly.plotly as py
import plotly.graph_objs as go
import time
import numpy as np
import cholesky
import lu
import tests
import scipy.linalg as sp

py.sign_in('kamp9', '09g4enb2lz')


def precision_test(repeat):
    test = []
    for i in range(repeat):
        rand_int_matrix = tests.generate_pos_dif(1000, 1000, 1000000)   # se int og real matrix
        decomp = sp.cholesky(rand_int_matrix)  # change function
        new_A = np.dot(decomp.transpose(), decomp)  # change function
        dif_matrix = new_A - rand_int_matrix
        test += [np.sum(np.abs(dif_matrix))]
        print i
    plot_data = [{
        'y': test,
        'type':'box',
        'marker':{'color': 'black'},
        'name': 'cholesky_out_of_place',
        'boxpoints': False
        }]
    url = py.plot(plot_data, filename='precision')

# precision_test(50)


def benchmark_test(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test = []
        for j in range(repeat):
            rand_int_matrix = np.random.randint(-1000, 1000, size=(i, i))
            time_start = time.clock()
            lu.lu_rook_pivot(rand_int_matrix)  # change funktion
            test += [time.clock() - time_start]
            print i, j
        plot_data += [{
            'y': test,
            'type':'box',
            'marker':{'color': 'black'},
            'name': str(i) + 'x' + str(i),
            'boxpoints': False
            }]
    url = py.plot(plot_data, filename='Benchmark')


benchmark_test(100, 1001, 50, 10)
