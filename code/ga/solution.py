""" Intializers for solution """

import numpy as np
from numpy import array, int64, float64
from . import utils

def get_random_binary_list_initializor(length, mu):
    def initialize_random_binary_list():

        sols = []
        for i in range(mu):
            sols.append( {'solution': np.random.randint(2, size=length, dtype=int)} )

        return sols

    return initialize_random_binary_list


def get_from_netshield(filename1, filename2, vertex_count):
    file_1 = open(filename1)
    cont_1 = file_1.read()
    sol_1 = eval(cont_1)

    file_2 = open(filename2)
    cont_2 = file_2.read()
    sol_2 = eval(cont_2)

    sol = utils.get_non_dominated(sol_1 + sol_2)

    for solution in sol:
        conv = np.zeros(vertex_count, dtype=bool)
        if solution['solution'].shape[0] == 0:
            solution['solution'] = conv
        else:
            conv[solution['solution']] = True
            solution['solution'] = conv


    def initialize_netshield():
        return sol

    return initialize_netshield