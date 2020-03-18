""" Objective functions for ga """

import numpy as np

def get_hamming_distance_objective(target1, target2):

    assert target1.shape == target2.shape, 'Target binary strings are of different length'

    def hamming_distance_objective(solution):

        assert solution['solution'].shape == target1.shape, \
            'Hamming distance binary strings are of different length'

        dist1 = target1.shape[0] - (solution['solution'] == target1).sum()
        dist2 = target2.shape[0] - (solution['solution'] == target2).sum()

        solution['evaluation'] = (dist1, dist2)

    return hamming_distance_objective


def _get_max_eigenvalue(adj, return_abs=True):
    eigvals = np.linalg.eigvalsh(adj)
    return np.abs(eigvals[-1])

def get_node_immunization_objectve(adj):
    original_eigval = _get_max_eigenvalue(adj)
    degrees = adj.sum(axis=0)

    def node_immunization_objective(solution):
        to_drop = solution['solution']

        ind = np.where(to_drop)
        adj_pert = np.array(adj)
        adj_pert[:,ind] = 0
        adj_pert[ind,:] = 0

        new_eigval = _get_max_eigenvalue(adj_pert)
        drop_eigval = original_eigval - new_eigval
        cost = (to_drop * degrees).sum()

        solution['evaluation'] = (-drop_eigval, cost)

    return node_immunization_objective




