"""
Netshield algorithm with greedy approximation replaced
with quadratic program solver.
Requires Gurobi + License.
http://www.gurobi.com/
"""

import numpy as np
from gurobipy import Model, GRB, QuadExpr, quicksum, LinExpr
import utils

def _is_dominated_by(A, B):
    """ Check if 2D solution A is dominated by B. First objective is maximised, second is minimised"""
    return (A[0] <= B[0] and A[1] > B[1]) or (A[0] < B[0] and A[1] >= B[1])

def _get_non_dominated(solutions):
    """ Filters out all dominated 2D solutions in the solution set and returns all nondominated ones """
    solutions.sort(key=lambda solution: (-solution['evaluation'][0], solution['evaluation'][1]) )
    pareto_front = []

    pareto_front.append(solutions[0])
    i = 0
    j = 1
    while j < len(solutions):
        if not _is_dominated_by(solutions[j]['evaluation'], solutions[i]['evaluation']):
            pareto_front.append(solutions[j])
            i = j
            j = i+1
        else:
            j += 1

    return pareto_front


def _build_objective_qd(adj, variables, eigval, eigvec):
    """ Build Shield-value objective function """
    obj = QuadExpr()
    eigvec_sq = np.square(eigvec)
    n = adj.shape[0]

    # Linear part
    for i in range(n):
        if eigvec_sq[i] != 0:
            obj.addTerms(2*eigval*eigvec_sq[i], variables[i])

    # Quadratic part
    for i in range(n):
        for j in range(i+1, n):
            if adj[i,j] != 0 and eigvec[i] != 0 and eigvec[j] != 0:
                obj.addTerms(-2 * adj[i,j] * eigvec[i] * eigvec[j], variables[i], variables[j])

    return obj
#

def _build_objective_linear(adj, variables, eigval, eigvec):
    """ Special case. Build linear objective function for QP. Used if batch size = 1 """
    obj = LinExpr()
    eigvec_sq = np.square(eigvec)
    n = adj.shape[0]

    for i in range(n):
        if eigvec_sq[i] != 0:
            obj.addTerms(2*eigval*eigvec_sq[i], variables[i])

    return obj


def _netshield_plus_mo_qp(adj, b, epsilon, context):
    """ Peforms actual NetShield algorithm """

    adj = np.array(adj)
    n = adj.shape[0]
    selected = np.zeros(n, dtype=bool)
    degrees = adj.sum(axis=0)
    m = context['model']
    m.setParam('OutputFlag', False)

    variables = context['variables']

    m.addConstr(context['degree_term'] <= epsilon)

    for i in range(1, int(np.ceil(n/b)) + 1):
        max_select = min(i*b, n)

        try:
            eigval, eigvec = context['computed_eigen'][selected.tobytes()]
        except KeyError:
            eigval, eigvec = utils.get_max_eigen(adj)
            eigvec[selected] = 0
            context['computed_eigen'][selected.tobytes()] = eigval, eigvec

        obj = context['obj_func'](adj, variables, eigval, eigvec)
        m.setObjective(obj, GRB.MAXIMIZE)

        add_variables = quicksum(variables)
        constr_add = m.addConstr(add_variables >= selected.sum()+1)
        constr_max = m.addConstr(add_variables <= max_select)

        m.optimize()

        if m.Status == GRB.INFEASIBLE:
            break

        for i, v in enumerate(m.getVars()):
            if v.x == 1 and not selected[i]:
                selected[i] = True
                m.addConstr(variables[i] == 1)

        m.remove(constr_add)
        m.remove(constr_max)

        adj[selected, :] = 0
        adj[:, selected] = 0


    m.remove(m.getConstrs())
    if selected.sum() == 0:
        return np.where(selected)[0], 0, 0

    eigval_pert = utils.get_max_eigenvalue(adj)
    drop = context['eigenvalue_ori'] - eigval_pert
    cost = degrees[selected].sum()

    return np.where(selected)[0], drop, cost


def netshield_plus_mo(adj, b, e_delta):
    """
    Perform the NetShield+ multiobjective algorithm via the epsilon constraint method.
    Epsilon values used range from 0 to sum of all degrees of the vertices in the input graph

    Parameters
    ----------
    A: 2D numpy array
        Adjancency matrix of graph to be immunised.
    b: Integer
        Batch size. How many nodes are removed in one step.
    e_delta: Integer
        By how much to increase the epsilon value after each step.

    Returns
    --------
    List of dictionaries that form the approximated Pareto front. Dictionaries have the following keys:
        solution: 1D numpy array
            indices of selectec vertices
        evaluation: tuple of (float,int)
            eigendrop, cost.

    """

    max_cost = adj.sum()
    solutions = []
    unique = set()
    n = adj.shape[0]
    degrees = adj.sum(axis=0)

    # Dict that contains needed that can be reused over all calls to netshield information
    # model: Gurobi QP model
    # obj_func: Gurobi objective function
    # variables: Gurobi variable objects
    # degree_term: Constraint of cost <= epsilon
    # computed_eigen: Contains all unique eigendecompositions for found solutions.
    #   Often the same (intermediate) solutions are found during the whole process
    #   This caches the result to avoid many same eigendecompositions that are computationally expensive.
    # eigenvalue_ori: Original eigenvalue of input graph

    context = {}

    if b == 1:
        context['model'] = Model('lp')
        context['obj_func'] = _build_objective_linear
    else:
        context['model'] = Model('qp')
        context['obj_func'] = _build_objective_qd

    context['variables'] = [context['model'].addVar(name="x_{}".format(i), vtype=GRB.BINARY)
                            for i in range(n)]

    context['degree_term'] = LinExpr()
    context['degree_term'].addTerms(degrees, context['variables'])
    context['computed_eigen'] = dict()
    context['eigenvalue_ori'] = utils.get_max_eigenvalue(adj)

    # Loop over all epsilon values
    for i in range(int(np.ceil(max_cost / e_delta)) + 1):
        epsilon = min(i * e_delta, max_cost)
        print('epsilon: ', epsilon)
        solution, drop, cost = _netshield_plus_mo_qp(adj, b, epsilon, context)

        if solution.tobytes() not in unique:
            solutions.append({'solution': solution,
                              'evaluation': (drop, cost)})
            unique.add(solution.tobytes())

    return _get_non_dominated(solutions)


def netshield_mo(adj, e_delta):
    """
    Perform the NetShield multiobjective algorithm via the epsilon constraint method.
    Epsilon values used range from 0 to sum of all degrees of the vertices in the input graph

    Parameters
    ----------
    A: 2D numpy array
        Adjancency matrix of graph to be immunised.
    e_delta: Integer
        By how much to increase the epsilon value after each step.

    Returns
    --------
    List of dictionaries that form the approximated Pareto front. Dictionaries have the following keys:
        solution: 1D numpy array
            indices of selectec vertices
        evaluation: tuple of (float,int)
            eigendrop, cost.

    """

    eigval, eigvec = utils.get_max_eigen(adj)
    degrees = adj.sum(axis=0)
    max_cost = degrees.sum()
    n = adj.shape[0]

    e_delta = min(max(e_delta, 1), max_cost)

    m = Model("qp")
    m.setParam('OutputFlag', False)

    variables = [m.addVar(name="x_{}".format(i), vtype=GRB.BINARY)
                 for i in range(n)]

    obj = _build_objective_qd(adj, variables, eigval, eigvec)

    constr = LinExpr()
    constr.addTerms(degrees, variables)
    m.setObjective(obj, GRB.MAXIMIZE)

    solutions = [{'solution': np.array([]), 'evaluation': (0, 0)}]
    unique = set()

    for i in range(int(np.ceil(max_cost / e_delta)) + 1):
        epsilon = min(i * e_delta, max_cost)
        print(epsilon)
        epsilon_constr = m.addConstr(constr <= epsilon, "c1")

        m.optimize()
        out = np.array([i for i, v in enumerate(m.getVars()) if v.x == 1])

        if out.shape[0] > 0 and out.tobytes() not in unique:
            adj_pert = np.array(adj)
            adj_pert[out, :] = 0
            adj_pert[:, out] = 0

            eig_drop = eigval - utils.get_max_eigenvalue(adj_pert)
            cost = degrees[out].sum()

            solution = {'solution': out,
                        'evaluation': (eig_drop, cost)}

            solutions.append(solution)
            unique.add(out.tobytes())

        m.remove(epsilon_constr)

    return _get_non_dominated(solutions)


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
        print(St)
        S[St] = True
        tot_eigdrop += eigdrop
        A_pert[S, :] = 0
        A_pert[:, S] = 0

    if k > t*b:
        St, eigdrop = netshield(A_pert, k-t*b, S)
        print(St)
        S[St] = True
        tot_eigdrop += eigdrop

    assert np.sum(S) == k, \
        "Incorrect number of nodes selected: {}, \
        expected: {}".format(np.sum(S), k)

    return np.where(S)[0], tot_eigdrop


def netshield(adj, k, al_out = None):
    """
    Perform the NetShield algorithm with QP solver

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

    eigval, eigvec = utils.get_max_eigen(adj)
    n = adj.shape[0]

    if k == 1:
        m = Model("lp")
    else:
        m = Model("qp")
    m.setParam('OutputFlag', False)

    variables = [m.addVar(name="x_{}".format(i), vtype=GRB.BINARY)
                 for i in range(n)]

    obj = _build_objective_qd(adj, variables, eigval, eigvec)

    const = quicksum(variables) == k

    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(const, "c1")

    if al_out is not None:
        for c in np.where(al_out)[0]:
            m.addConstr(variables[c] == 0)

    m.optimize()

    out = np.array([i for i, v in enumerate(m.getVars()) if v.x == 1])

    adj_pert = np.array(adj)
    adj_pert[out, :] = 0
    adj_pert[:, out] = 0

    eig_drop = eigval - utils.get_max_eigenvalue(adj_pert)

    return out, eig_drop