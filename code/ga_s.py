""" Genetic algorithm for node immunization problem """

import numpy as np
import utils
from itertools import product


def eigen_drop(matrix, sol, lambd):
    """ Fitness function """
    matrix = np.array(matrix)
    matrix[sol, :] = 0
    matrix[:, sol] = 0

    eigval_pert = utils.get_max_eigenvalue(matrix)

    return lambd - eigval_pert


def fitness_proportionate_select(population, fitness):
    prob = np.cumsum(fitness/sum(fitness))
    bound = np.random.uniform()
    ind = (bound < prob).argmax()
    return population[ind, :]


def recombine(parent_1, parent_2, k):
    union = np.union1d(parent_1[:k], parent_2[:k])
    if union.shape[0] > k:
        to_keep = np.random.choice(union.shape[0], size=k, replace=False)
        union = union[to_keep]

    complement = np.setdiff1d(np.arange(parent_1.shape[0]), union)
    new = np.concatenate((union, complement))

    # assert len(np.unique(new)) == new.shape[0] and np.all(new < new.shape[0])
    # and np.all(new >= 0), "Invalid solution created during recombination"
    return new


def cross_over(population, fitness, k, pc):
    parent_1 = fitness_proportionate_select(population, fitness)

    if np.random.uniform() < pc:
        parent_2 = fitness_proportionate_select(population, fitness)
        return recombine(parent_1, parent_2, k)

    return np.array(parent_1)


def mutate(new, pm, p, u_ind_topk):
    n = new.shape[0]
    k = u_ind_topk.shape[0]

    prob = np.repeat(pm, n)
    prob[np.in1d(new, u_ind_topk)] = p

    p_in = prob[:k] / np.sum(prob[:k])
    p_out = prob[k:] / np.sum(prob[k:])

    for _ in range(k):
        ind_in = np.random.choice(k, size=1, replace=False, p=p_in)
        ind_out = k + np.random.choice(n-k, size=1, replace=False, p=p_out)
        new[ind_in], new[ind_out] = new[ind_out], new[ind_in]

        # prob[ind_in], prob[ind_out] = prob[ind_out], prob[ind_in]
        # p_in = prob[:k] / np.sum(prob[:k])
        # p_out = prob[k:] / np.sum(prob[k:])

    return new


def ga_i(graph, k, prob_top, max_evals):
    """
    Genetic Algorithm for k-node immunisation problem

    Parameters
    ----------
    graph: igraph graph object
		Graph to immunise.
    k:	Integer
        Size of solution set k
    prob_top: float
        Rate at which top k eigenscore nodes are chosen during mutation
    max_evals: Integer
        Maximum number of generations.

    Returns
    --------
    Tuple of numpy array, a float and an integer
        First is the solution set containing indices of selected nodes
        Second is the fitness eigendrop
		Third is the number of generations. Either max_evals or smaller when the early
		stopping criterion is hit.

    Notes
    ------
    A simulation is run for every possible combination of lambds, mus
    and max_steps.
    """


    # GA parameters
    n = graph.vcount()
    mu = 50
    pc = 1
    pm = 1/n

    matrix = utils.get_adj_np(graph)
    lambd, u = utils.get_max_eigen(matrix)
    u_ind_topk = u.argsort()[::-1][:k]

    population = np.zeros((mu, n), dtype=np.int)
    for i in range(mu):
        population[i, :] = np.random.choice(n, size=n, replace=False)

        # assert len(np.unique(P[i,:])) == n and np.all(P[i,:] < n)
        # and np.all(P[i,:] >= 0), "Invalid solution generated"

    fitness = np.zeros(mu)
    for i in range(mu):
        fitness[i] = eigen_drop(matrix, population[i, :k], lambd)

    hist_best = [np.max(fitness)]
    stagnation_count = 0

    gen_count = 0
    for gen_count in range(1, max_evals):
        #print(gen_count)
        population_new = np.zeros((mu, n), dtype=int)
        fitness_new = np.zeros(mu)

        for i in range(mu):
            offspring = cross_over(population, fitness, k, pc)
            mutated_offspring = mutate(offspring, pm, prob_top, u_ind_topk)

            population_new[i, :] = mutated_offspring
            fitness_new[i] = eigen_drop(matrix, mutated_offspring[:k], lambd)

        population_both = np.vstack((population, population_new))
        fitness_both = np.concatenate((fitness, fitness_new))

        ind_best = fitness_both.argsort()[::-1][:mu]

        population = population_both[ind_best]
        fitness = fitness_both[ind_best]

        fit_best = np.max(fitness)
        hist_best.append(fit_best)

        if (gen_count-1) % 100 == 0:
            print("generation: {}, best fitness: {}".format(gen_count, fit_best))

        if hist_best[-1] == hist_best[-2]:
            stagnation_count += 1
            if stagnation_count > k * (n-k):
                break

        else:
            stagnation_count = 0

    ind_best = fitness.argmax()
    return population[ind_best, :k], fitness[ind_best], gen_count


def ga_i_m(graph, k, prob_top, max_evals):
    """
    Perform multiple genetic algorithms and return results.

    Parameters
    ----------
    graph: igraph graph object
		Graph to immunise.
    k:	List of integers
        Sizes of solution set k
    prob_top: List of floats
        Rates at which top k eigenscore nodes are chosen during mutation
    max_evals: List of integer
        Maximum number of generations.

    Returns
    --------
    List of tuple of numpy array, a float and an integer containing results
        First is the solution set containing indices of selected nodes
        Second is the fitness eigendrop
		Third is the number of generations. Either max_evals or smaller when the early
		stopping criterion is hit.

    Notes
    ------
    A GA is run for every combination of k, prob_top and max_evals.
    """

    confs = product(k, prob_top, max_evals)

    results = []
    for conf in confs:
        result = ga_i(graph, *conf)
        results.append(result)

    return results;