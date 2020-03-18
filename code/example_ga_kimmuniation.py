"""
Runs k-immunisation GA 6 times for different top eigenscore selection rates
Outputs results in terms of eigendrop together with NetShield+ eigendrop result
"""

import utils
import ga_s
import netshield

graph = utils.get_graph("graphs/karate.gml")
n = len(graph.vs)
k = 5
max_evals = 1000
rates = [1/n, 2/n, 3/n, 6/n, 1]
fitnesses = []

for rate in rates:
    print("Running GA with top eigenscore selection rate {}".format(rate))
    sol, eigdrop, gen = ga_s.ga_i(graph, k, rate, max_evals)
    fitnesses.append(eigdrop)

for i in range(0, len(rates)):
    print("Rate {}: {}".format(rates[i], fitnesses[i]))


adj = utils.get_adj_np(graph)
sol, eigdrop = netshield.netshield_plus(adj, k, 1)

print("NetShield+ {}".format(eigdrop))
