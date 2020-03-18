import numpy as np
import utils

"""" Implements standard NetShield node immunisation algorithms """

def netshield(A, k, al_out = None):
    """
    Perform the NetShield algorithm

    Parameters
    ----------
    A: 2D numpy array
        Adjancency matrix of graph to be immunised
    k: Integer
        Number of nodes to remove from graph
    al_out: Integer
        Optional, already removed nodes or nodes that should be ignored. Not intended for direct callers
    
        
    Returns
    --------
    Tuple of 1D numpy array and float
        First is indices of selected nodes
        Second is eigendrop
    """

    n = A.shape[0]

    shieldvalue = 0

    if(k < 1):
        return np.array([], dtype=np.int64), 0

    lambd, u = utils.get_max_eigen(A)

    # Compute individual Shield score of each node
    v = (2 * lambd - A.diagonal()) * np.square(u)

    n = A.shape[0]
    S = np.zeros(n, dtype=bool)

    # Add vertices greedily
    for _ in range(k):
        B = A[:, S]
        b = B.dot(u[S])
        score = v - (2 * b * u)
        score[S] = -1

        if al_out is not None:
            score[al_out] = -1

        i = np.argmax(score)
        S[i] = True
        shieldvalue += score[i]

    A_pert = np.array(A)
    A_pert[S, :] = 0
    A_pert[:, S] = 0

    eigdrop = lambd - utils.get_max_eigenvalue(A_pert)

    return np.where(S)[0], eigdrop


def netshield_plus(A, k, b):
    """
    Perform the NetShield+ algorithm

    Parameters
    ----------
    A: 2D numpy array
        Adjancency matrix of graph to be immunised.
    k: Integer
        Number of nodes to remove from graph.
    b: Integer
        Batch size < k to be removed each step.
        
    Returns
    --------
    Tuple of 1D numpy array and float
        First is indices of selected nodes
        Second is eigendrop
    """

    if(k < 1):
        return np.array([], dtype=np.int64), 0

    A_pert = np.array(A)
    n = A.shape[0]
    S = np.zeros(n, dtype=bool)
    tot_eigdrop = 0

    t = int(k/b)

    for _ in range(t):
        (St, eigdrop) = netshield(A_pert, b, S)
        S[St] = True
        tot_eigdrop += eigdrop
        A_pert[S, :] = 0
        A_pert[:, S] = 0

    if k > t*b:
        St, eigdrop = netshield(A_pert, k-t*b, S)
        S[St] = True
        tot_eigdrop += eigdrop

    assert np.sum(S) == k, \
        "Incorrect number of nodes selected: {}, \
        expected: {}".format(np.sum(S), k)

    return np.where(S)[0], tot_eigdrop
