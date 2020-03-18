"""
Uses NetShield and NetShield+ to find immunisation solution
Then performs a single CTMC simulation with the SIS model
for original and both immunised networks
The number of infected nodes is plotted against the timesteps
"""

import utils
import netshield
import ctmc
import numpy as np
import matplotlib.pyplot as plt


name = "graphs/day3.gml"
k = 20
b = 1
lambd = 0.25
mu = 0.1
max_steps = 1000

graph = utils.get_graph(name)
adj = utils.get_adj_np(graph)


res_ns = netshield.netshield(adj, k)

print("Result for NetShield algorithm: Indices: {}, eigendrop: {}".format(res_ns[0], res_ns[1]))

res_ns_plus = netshield.netshield_plus(adj, k, b)

print("Result for NetShield+ algorithm with batch size 1: Indices: {}, eigendrop: {}".format(res_ns_plus[0], res_ns_plus[1]))


infected, time_passed = ctmc.ctmc_sis(graph, lambd, mu, max_steps)

adj_ns = np.array(adj)
adj_ns[res_ns[0], :] = 0
adj_ns[:, res_ns[0]] = 0

graph_ns = utils.get_graph_np(adj_ns)

infected_ns, time_passed_ns = ctmc.ctmc_sis(graph_ns, lambd, mu, max_steps)

adj_ns_plus = np.array(adj)
adj_ns_plus[res_ns_plus[0], :] = 0
adj_ns_plus[:, res_ns_plus[0]] = 0

graph_ns_plus = utils.get_graph_np(adj_ns_plus)

infected_ns_plus, time_passed_ns_plus = ctmc.ctmc_sis(graph_ns_plus, lambd, mu, max_steps)


plt.plot(range(len(infected)), infected, label="Original")
plt.plot(range(len(infected_ns)), infected_ns, label="NetShield Immunisation")
plt.plot(range(len(infected_ns_plus)), infected_ns_plus, label="NetShield+ Immunisation")

plt.xlabel("Time step")
plt.ylabel("Nodes Infected")
plt.title("Infected nodes at time steps")
plt.legend()

plt.show()