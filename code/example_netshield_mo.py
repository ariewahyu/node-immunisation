"""
Approximates Pareto front for multiobjective immunisation
with the NetShield and NetShield+ methods with epsilon constraint
Then plots the results
"""

import utils
import netshield_qp
import matplotlib.pyplot as plt

graph = utils.get_graph("graphs/karate.gml")
adj = utils.get_adj_np(graph)

res_netshield = netshield_qp.netshield_mo(adj, 1)
res_netshield_plus = netshield_qp.netshield_plus_mo(adj, 1, 1)

# Plotting the Pareto front

coordinates_netshield = ( s['evaluation'] for s in res_netshield)
xy = list(zip(*coordinates_netshield))
plt.plot(xy[0], xy[1], 'ro', label="NetShield")

coordinates_netshield_plus = ( s['evaluation'] for s in res_netshield_plus)
xy = list(zip(*coordinates_netshield_plus))
plt.plot(xy[0], xy[1], 'bo', label="NetShield+")

plt.title("Pareto front")
plt.xlabel("eigendrop")
plt.ylabel("cost")
plt.legend()
plt.show()