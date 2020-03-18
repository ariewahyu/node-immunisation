"""
Monte carlo sampling of n graph some v. Both parameters should be passed as command line argument
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
from igraph import Graph
import matplotlib.pyplot as plt
import os
import sys
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
        Sv_q += adj[i, j] * u[i] * u[j]

    return Sv_p - Sv_q


if len(sys.argv) < 3:
    print("Usage: python monte_carlo.py <vertices> <nrsamples>")
    exit(-1)

nrvertex = int(sys.argv[1])
number = int(sys.argv[2])

dirname = "monte_carlo_v{}".format(nrvertex)

i = 0
while True:
    if not os.path.exists("{}_{}".format(dirname, i)):
        dirname = "{}_{}".format(dirname, i)
        os.makedirs(dirname)
        break
    else:
        i += 1


print("Storing results in: ", dirname)

eigvals = []
measures = [ {"func": lambda graph: Graph.average_path_length(graph, unconn = False), "result" : [], "name" : "Average Path Length"},
             {"func": lambda graph: sum(Graph.degree(graph))/len(Graph.degree(graph)), "result" : [], "name" : "Average Degree"},
             #{"func": lambda graph: max(Graph.degree(graph)), "result" : [], "name" : "Maximum Degree"},
             {"func": lambda graph: sum(Graph.betweenness(graph))/len(Graph.betweenness(graph)), "result" : [], "name" : "Average Betweenness"},
             #{"func": lambda graph: max(Graph.betweenness(graph)), "result" : [], "name" : "Maximum Betweenness"},
             {"func": lambda graph: len(Graph.cliques(graph)), "result" : [], "name" : "Number of Cliques"},
             #{"func": lambda graph: Graph.clique_number(graph), "result" : [], "name" : "Largest Clique"},
             {"func": lambda graph: Graph.transitivity_undirected(graph, mode="zero"), "result" : [], "name" : "Global Clustering Coefficient"},
             #{"func": lambda graph: Graph.transitivity_avglocal_undirected(graph, mode="nan"), "result" : [], "name" : "Averega Local Clustering Coefficient"},
             {"func": lambda graph: shieldvalue(list(range(0,nrvertex)), graph), "result" : [], "name" : "Shield Value"},]

for i in range(0, number):
    print("Graph: ", i)
    graph = Graph.Erdos_Renyi(nrvertex, 0.5)
    adj = np.array(graph.get_adjacency()._get_data())
    eigv = np.linalg.eigvals(adj)
    x = np.real(max(eigv))
    eigvals.append(x)
    for measure in measures:
        measure["result"].append(measure["func"](graph))

for measure in measures:
    (pearson, pv) = scipy.stats.pearsonr(measure["result"], eigvals)
    print(measure["name"], pearson)
    plt.plot(measure["result"], eigvals, 'o')
    plt.ylabel("Max eigen value")
    plt.xlabel(measure["name"])
    plt.title(measure["name"])
    plt.savefig(dirname + "/" + measure["name"] + ".png", dpi=200)
    plt.clf()



