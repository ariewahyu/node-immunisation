""" Several crossover operators """

import numpy as np

def check_arguments(p1, p2, N=None):
    assert p1['solution'].shape == p2['solution'].shape, \
        'Parent solutions need to have the same length {} != {}' \
        .format(p1['solution'].shape, p2['solution'].shape)

    if N:
        assert p1['solution'].shape[0] >= N, \
            'More crossoverpoints than solution lenghts {} > {}' \
            .format(p1['solution'].shape[0], N)


def get_npoint_crossover(N, pc=1):

    def npoint_crossover(p1, p2, crossover_points=None):
        check_arguments(p1, p2, N)

        if(np.random.random() > pc): return {'solution': np.array(p1['solution'])}

        length = p1['solution'].shape[0]

        if crossover_points is None:
            crossover_points = np.random.choice(length, N, replace=False)

        crossover_points.sort()
        crossover_points = np.hstack((0, crossover_points, length))

        child = np.zeros(length, dtype=p1['solution'].dtype)
        for i, _ in enumerate(crossover_points[:-1]):
            child[crossover_points[i]:crossover_points[i+1]] = \
                p1['solution'][crossover_points[i]:crossover_points[i+1]]

            (p1, p2) = (p2, p1)

        return {'solution': child}

    return npoint_crossover


def get_uniform_crossover(pc=1):

    def uniform_crossover(p1, p2):
        check_arguments(p1, p2)

        if(np.random.random() > pc): return {'solution': np.array(p1['solution'])}

        child = np.array([bits[i%2] for i, bits in enumerate(zip(p1['solution'], p2['solution']))],
                         dtype=p1['solution'].dtype)
        return {'solution': child}

    return uniform_crossover
