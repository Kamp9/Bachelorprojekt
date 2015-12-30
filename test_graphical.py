import plotly.plotly as py
import time
import numpy as np
import cholesky
import lu

py.sign_in('kamp9', '09g4enb2lz')


def benchmark(minsize, maxsize, step, repeat):
    plot_data = []
    for i in range(minsize, maxsize, step):
        test = []
        for j in range(repeat):
            rand_int_matrix = np.random.randint(-1000, 1000, size=(i, i))
            time_start = time.clock()
            lu.lu_in_place(rand_int_matrix)  # change funktion
            test += [time.clock() - time_start]
        plot_data += [{
            'y': test,
            'type':'box',
            'marker':{'color': 'black'},
            'name': ' ' + str(i),
            }]
        print i
    url = py.plot(plot_data, filename='Benchmark')


benchmark(900, 1000, 1, 1)
