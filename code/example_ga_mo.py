"""
Approximates Pareto front for multiobjective immunisation
with a single run of SMS-EMOA and NSGAII GA
Then plots the results
"""

import ga
import utils
import matplotlib.pyplot as plt

graph = utils.get_graph("graphs/karate.gml")
adj = utils.get_adj_np(graph)

# The GA operators
parameters = {}

parameters['lda'] = 1 #lambda
parameters['evaluations'] = 10000

#intializes 100 random binary vector solutions of size |V|
parameters['solution_initializer'] = ga.solution.get_random_binary_list_initializor(graph.vcount(), 100)

# Node immunisation objective: Eigendrop and Cost.
# GAs work on all minimisation, so eigendrop will multiplied by -1 during optimization
parameters['objective_function'] = ga.objectives.get_node_immunization_objectve(adj)

# Uniform mutation, every bit flips with probability of 1/|V|
parameters['mutator'] = ga.mutation.get_uniform_mutator(1/graph.vcount())

# Parent selection is random
parameters['parent_selector'] = ga.parent_selection.get_random_selector()

# 1-point crossover with probability of 0.75
parameters['crossover'] = ga.crossover.get_npoint_crossover(1, 0.75)

# First SMS-EMOA

# Rank via the SMS-EMOA method. Reference point is (1, sum of all degrees)
ranker = ga.ranking.get_SMSEMOA_ranker((1, sum(graph.degree())+1))

# mu+lambda selection scheme (here 100+1)
parameters['selector'] = ga.selection.get_from_all(ranker)


result_smsemoa = ga.algorithm.genetic_algorithm(**parameters)

# Then NSGAII

# Different ranker
ranker = ga.ranking.get_NSGAII_ranker()

# Selection scheme remains the same
parameters['selector'] = ga.selection.get_from_all(ranker)

result_nsgaii = ga.algorithm.genetic_algorithm(**parameters)

coordinates = ((-solution['evaluation'][0], solution['evaluation'][1])  for solution in result_smsemoa)
xy = list(zip(*coordinates))
plt.plot(xy[0], xy[1], 'ro', label="SMS-EMOA")


coordinates = ((-solution['evaluation'][0], solution['evaluation'][1])  for solution in result_nsgaii)
xy = list(zip(*coordinates))
plt.plot(xy[0], xy[1], 'bo', label="NSGAII")

plt.title("SMS-EMOA vs NSGAII")
plt.xlabel("Eigendrop")
plt.ylabel("Cost")
plt.legend()
plt.show()


