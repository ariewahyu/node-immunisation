"""
Fully enumerates over all possible graphs with |V|=5.
Computes a set of measures:
- Average path length
- Average degree
- Average betweenness
- Number of cliques
- Global clustering coefficient
- Shield value
Plots the results for all measures against the eigenvalue of the graph
"""


import itertools
import numpy as np
import igraph
from igraph import Graph
import matplotlib.pyplot as plt
import scipy.stats


def shieldvalue(S, graph):
    adj = np.array(graph.get_adjacency().data)
    eigvals, eigvecs = np.linalg.eig(adj)
    max_eig_ind = np.argmax(np.real(eigvals))

    lambd = np.real(eigvals[max_eig_ind])
    u = np.real(eigvecs[:, max_eig_ind])

    Sv_p = 0
    for i in S:
        Sv_p += 2*lambd*np.square(u[i])

    Sv_q = 0
    for i, j in itertools.combinations(S, 2):
        Sv_q += 2 * adj[i, j] * u[i] * u[j]

    return Sv_p - Sv_q


def EnumerateAdjacency(dim):
    mat = [[0 for i in range(0, dim)] for i in range(0, dim)]
    yield from CreateReal(1, dim, mat)


def CreateReal(start, end, mat):
    if end-start > 0:
        for i in itertools.product([0, 1], repeat=(end-start)):
            mat[start-1][start:end] = i
            yield from CreateReal(start+1, end, mat)
    else:
        yield mat


def do_enumeration():
    dim = 5

    eigvals = []
    #  List of measures.
    #  The dict key "func" should be a function that takes a graph and returns a measure result.
    measures = [{"func": lambda graph: Graph.average_path_length(graph, unconn=False), "result": [], "name": "Average Path Length"},
                {"func": lambda graph: sum(Graph.degree(graph))/len(Graph.degree(graph)), "result": [], "name": "Average Degree"},
                # {"func": lambda graph: max(Graph.degree(graph)), "result" : [], "name" : "Maximum Degree"},
                {"func": lambda graph: sum(Graph.betweenness(graph))/len(Graph.betweenness(graph)), "result": [], "name": "Average Betweenness"},
                # {"func": lambda graph: max(Graph.betweenness(graph)), "result" : [], "name" : "Maximum Betweenness"},
                {"func": lambda graph: len(Graph.cliques(graph)), "result": [], "name": "Number of Cliques"},
                # {"func": lambda graph: Graph.clique_number(graph), "result" : [], "name" : "Largest Clique"},
                {"func": lambda graph: Graph.transitivity_undirected(graph, mode="zero"), "result": [], "name": "Global Clustering Coefficient"},
                # {"func": lambda graph: Graph.transitivity_avglocal_undirected(graph, mode="nan"), "result" : [], "name" : "Average Local Clustering Coefficient"},
                {"func": lambda graph: shieldvalue(list(range(0,dim)), graph), "result": [], "name": "Shield Value"}]

    for i in EnumerateAdjacency(dim):
        upper = np.array(i)
        adj = np.maximum(upper, np.transpose(upper))
        eigv = np.linalg.eigvals(adj)
        x = np.real(max(eigv))
        graph = Graph.Adjacency(adj.tolist(), igraph.ADJ_UNDIRECTED)
        eigvals.append(x)
        for measure in measures:
            measure["result"].append(measure["func"](graph))

    for measure in measures:
        (pearson, _) = scipy.stats.pearsonr(measure["result"], eigvals)
        print(measure["name"], pearson)
        plt.plot(measure["result"], eigvals, 'o')
        plt.ylabel("Max eigen value")
        plt.xlabel(measure["name"])
        plt.title(measure["name"])
        plt.savefig(measure["name"] + ".png", dpi=200)
        plt.clf()



if __name__ == "__main__":
    do_enumeration()

