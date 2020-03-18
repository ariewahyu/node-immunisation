"""
For performing continous time markov chain simulations of graph infections.
Both SI and SIS models available.
Multiple simulations can be run simultaneously via the multiprocessing package.
Requires igraph and numpy packages
"""

from math import inf
from multiprocessing import Pool
from itertools import product
from datetime import datetime
from igraph import Graph
import numpy as np


def _roulette_selection(probabilities):
    """
    Perform roulette wheel selection with the supplied probabilities

    Parameters
    ----------
    probabilities: numpy array
        Proportionate probabilities to select from. Should sum to 1.

    Returns
    --------
    int
        index of selected probability.
    """

    prob_cumsum = np.cumsum(probabilities)
    bound = np.random.uniform()
    return (bound < prob_cumsum).argmax()


def _ctmc_next_step_si(graph, states, lambd):
    """
    Perform the next time step of the SI model.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    states: binary numpy array
        State of each vertex.
    lambd: float
        Infection contagiousness rate.

    Returns
    -------
    int
        Next infected vertex
    float
        Delta time until next event.
        Will be inf if absorbing state has been reached.
    """

    rates_i = np.zeros(graph.vcount())
    for j in np.where(np.invert(states))[0]:
        rates_i[j] = lambd * states[graph.neighbors(j)].sum()

    rates_ii = rates_i.sum()
    if rates_ii == 0:
        return (-1, inf)

    new_infected = _roulette_selection(rates_i/rates_ii)
    delta_time = np.random.exponential(1/rates_ii)

    return new_infected, delta_time


def ctmc_si(graph, lambd, max_steps):
    """
    Peform a continuous time markov chain simulation based on an SI model
    (Susceptible-Infected).

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambd: float
        Infection contagiousness rate.
    max_steps: int
        Maximum number of steps to run the simulation for.
        Simulation stop early if absorbing state is reached.

    Returns:
    --------
    list
        Number of infected vertices after each step.
    list
        Total time passed after each simulation step.
    """

    vertex_count = graph.vcount()
    states = np.zeros(vertex_count, dtype=bool)

    initial_infected = np.random.randint(vertex_count)
    states[initial_infected] = True

    nr_infected = [1]
    time_passed = [0.0]

    for _ in range(1, max_steps):
        new_infected, delta_time = _ctmc_next_step_si(graph, states, lambd)

        if delta_time == inf:
            break

        states[new_infected] = True

        nr_infected.append(nr_infected[-1] + 1)
        time_passed.append(time_passed[-1] + delta_time)

    return nr_infected, time_passed


def _ctmc_next_step_sis(graph, states, lambd, mu):
    """
    Perform the next time step of the SIS model.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    states: binary numpy array
        State of each vertex.
    lambd: float
        Infection contagiousness rate.
    mu: float
        Rate at which infected vertices become susceptible again

    Returns
    -------
    int
        Next infected vertex
    float
        Delta time until next event
        Will be inf if absorbing state has been reached.
    """

    rates_i = np.zeros(graph.vcount())
    for j, state in enumerate(states):
        if state:
            rates_i[j] = mu
        else:
            rates_i[j] = lambd * states[graph.neighbors(j)].sum()

    rates_ii = rates_i.sum()
    if rates_ii == 0:
        return(-1, inf)

    changed = _roulette_selection(rates_i/rates_ii)
    delta_time = np.random.exponential(1/rates_ii)

    return changed, delta_time


def ctmc_sis(graph, lambd, mu, max_steps):
    """
    Peform a continuous time markov chain simulation based on an SIS model
    (Susceptible-Infected-Susceptible).

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambd: float
        Infection contagiousness rate.
    mu: float
        Rate at which infected vertices become susceptible again
    max_steps: int
        Maximum number of steps to run the simulation for.
        Simulation stop early if absorbing state is reached.

    Returns:
    --------
    list
        Number of infected vertices after each step.
    list
        Total time passed after each simulation step.
    """

    vertex_count = graph.vcount()
    states = np.zeros(vertex_count, dtype=bool)

    initial_infected = np.random.randint(vertex_count)
    states[initial_infected] = True

    nr_infected = [1]
    time_passed = [0.0]

    for _ in range(1, max_steps):
        changed, delta_time = _ctmc_next_step_sis(graph, states, lambd, mu)

        if delta_time == inf:
            break

        states[changed] = not states[changed]
        nr_infected.append(nr_infected[-1] + (1 if states[changed] else -1))
        time_passed.append(time_passed[-1] + delta_time)

    return nr_infected, time_passed


class _CmtcMpWrapper:
    def __init__(self, graph, func):
        self.graph = graph
        self.func = func

    def __call__(self, *args):
        return self.func(self.graph, *args)


def ctmc_si_multiple_mp(graph, lambds, max_steps):
    """
    Perform multiple continuous time markov chain simulations
    with SI model in parallel.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambds: list of floats
        List of infection contagiousness rates.
    max_steps: list of ints
        List of max time steps.

    Returns
    --------
    list of tuples of two lists
        First is number of infected vertices after each step
        Second is time passed after each step

    Notes
    ------
    A simulation is run for every possible combination of lambds and max_steps.
    """

    pool = Pool()
    func = _CmtcMpWrapper(graph, ctmc_si)
    confs = product(lambds, max_steps)
    results = []
    for run_result in pool.starmap(func, confs):
        results.append(run_result)
        # print("Total time: ", time_passed[-1], "infected: ", nr_infected[-1])

    return results


def ctmc_si_multiple(graph, lambds, max_steps):
    """
    Perform multiple continuous time markov chain simulations
    with SI model sequentially.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambds: list of floats
        List of infection contagiousness rates.
    max_steps: list of ints
        List of max time steps.

    Returns
    --------
    list of tuples of two lists
        First is number of infected vertices after each step
        Second is time passed after each step

    Notes
    ------
    A simulation is run for every possible combination of lambds and max_steps.
    """

    confs = product(lambds, max_steps)
    results = []
    for conf in confs:
        run_result = ctmc_si(graph, *conf)
        results.append(run_result)
        # print("Total time: ", time_passed[-1], "infected: ", nr_infected[-1])

    return results


def ctmc_sis_multiple_mp(graph, lambds, mus, max_steps):
    """
    Perform multiple continuous time markov chain simulations
    with SIS model in parallel.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambds: list of floats
        List of infection contagiousness rates.
    mus: list of floats
        List of rates at which vertices become susceptible again
    max_steps: list of ints
        List of max time steps.

    Returns
    --------
    list of tuples of two lists
        First is number of infected vertices after each step
        Second is time passed after each step

    Notes
    ------
    A simulation is run for every possible combination of lambds, mus
    and max_steps.
    """

    pool = Pool()
    func = _CmtcMpWrapper(graph, ctmc_sis)
    confs = product(lambds, mus, max_steps)
    results = []
    for run_result in pool.starmap(func, confs):
        results.append(run_result)
        # print("Total time: ", time_passed[-1], "infected: ", nr_infected[-1])

    return results


def ctmc_sis_multiple(graph, lambds, mus, max_steps):
    """
    Perform multiple continuous time markov chain simulations
    with SIS model sequentially.

    Parameters
    ----------
    graph: igraph graph object
        Graph on which the simulation is run.
    lambds: list of floats
        List of infection contagiousness rates.
    mus: list of floats
        List of rates at which vertices become susceptible again
    max_steps: list of ints
        List of max time steps.

    Returns
    --------
    list of tuples of two lists
        First is number of infected vertices after each step
        Second is time passed after each step

    Notes
    ------
    A simulation is run for every possible combination of lambds, mus
    and max_steps.
    """

    confs = product(lambds, mus, max_steps)
    results = []
    for conf in confs:
        run_result = ctmc_sis(graph, *conf)
        results.append(run_result)
        # print("Total time: ", time_passed[-1], "infected: ", nr_infected[-1])

    return results


def _do_test():
    vertex_count = 50
    max_steps = [vertex_count*2]
    graph = Graph.Erdos_Renyi(vertex_count, 0.5)

    lambds = [0.01, 0.02, 0.04, 0.08]
    mus = [0.01, 0.02, 0.04, 0.08]

    start_time = datetime.now()
    ctmc_sis_multiple(graph, lambds, mus, max_steps)
    print("Time passed: ", datetime.now() - start_time)


if __name__ == '__main__':
    _do_test()
