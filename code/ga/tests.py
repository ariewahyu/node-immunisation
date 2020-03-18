import numpy as np
from . import crossover
from . import mutation
from . import parent_selection
from . import ranking
from . import selection
from . import solution
from . import objectives
from itertools import product
from math import inf


def test_hamming_objectives_targets_length_differs():
    t1 = np.array([1,0,0,1])
    t2 = np.array([1,0,0,1,1])

    try:
        objectives.get_hamming_distance_objective(t1,t2)
    except AssertionError as ex:
        assert str(ex) == 'Target binary strings are of different length', 'Wrong exception string'

def test_hamming_objectives_target_and_solution_length_differs():
    t1 = np.array([1,0,0,0,1])
    t2 = np.array([1,0,1,0,1])
    s = {'solution': np.array([1,0,0,1])}

    objective = objectives.get_hamming_distance_objective(t1,t2)

    try:
        objective(s)
    except AssertionError as ex:
        assert str(ex) == 'Hamming distance binary strings are of different length', \
            'Wrong exception string'

def test_hamming_objectives1():
    t1 = np.array([1,0,1,0,1])
    t2 = np.array([0,0,0,1,1])
    s =  {'solution': np.array([1,1,0,0,1])}

    expected = (2,3)
    objective = objectives.get_hamming_distance_objective(t1,t2)
    objective(s)

    assert expected == s['evaluation'], \
        'Expected objective results do not match actual results {} {}' \
        .format(expected, s['evaluation'])


def test_uniform_mutatorP05():
    prob = 0.5
    mutator = mutation.get_uniform_mutator(prob)
    s = {'solution': np.random.randint(2, size=10, dtype=int)}
    seed = 0
    np.random.seed(seed)
    generated = np.random.random(10) < prob
    np.random.seed(seed)
    m = mutator(s)

    for i in range(0,10):
        if(generated[i]):
            assert m['solution'][i] != s['solution'][i], "Should have flipped"
        else:
            assert m['solution'][i] == s['solution'][i], "Should not have flipped"


def test_uniform_mutatorP1():
    mutator = mutation.get_uniform_mutator(1)
    s = {'solution': np.zeros(0, dtype=int)}
    m = mutator(s)

    assert (m['solution'] == 1).all(), "All bits of mutated solution should be 1"

def test_uniform_mutatorP0():
    mutator = mutation.get_uniform_mutator(1)
    s = {'solution': np.zeros(0, dtype=int)}
    m = mutator(s)

    assert (m['solution'] == 0).all(), "All bits of mutated solution should be 1"


def test_SMSEMOA_ranker():
    reference_point = (5,5)
    eval_solutions = [{'solution': 3, 'evaluation': (2,2)},
                      {'solution': 4, 'evaluation': (4,3)},
                      {'solution': 1, 'evaluation': (3,1)},
                      {'solution': 2, 'evaluation': (1,4)},
                      {'solution': 4, 'evaluation': (3,4)},
                      {'solution': 3, 'evaluation': (2,2)}]


    expected_ranking = [{'solution': 1, 'evaluation': (3,1), 'rank': 0, 'score': 2},
                        {'solution': 2, 'evaluation': (1,4), 'rank': 0, 'score': 1},
                        {'solution': 3, 'evaluation': (2,2), 'rank': 0, 'score': 0},
                        {'solution': 3, 'evaluation': (2,2), 'rank': 0, 'score': 0},
                        {'solution': 4, 'evaluation': (3,4), 'rank': 1, 'score': 1},
                        {'solution': 4, 'evaluation': (4,3), 'rank': 1, 'score': 1}]


    eval_solutions_added = [(0,0),
                          (1,1),
                          (0,2),
                          (0,1),
                          (1,1),
                          (0,0)]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)

    out = ranker(eval_solutions)

    assert len(expected_ranking) == len(out), \
        "Solutions got lost: e: {}, r: {}".format(len(expected_ranking), len(out))

    for exp,real in zip(expected_ranking, out):
        assert (exp['solution'] == real['solution'] and exp['rank'] == real['rank']
            and exp['score'] == real['score']), \
            "Invalid ranking, expected:{}, real:{}".format(exp, real)

    for eval_sol, added in zip(eval_solutions,eval_solutions_added):
        assert eval_sol['rank'] == added[0] and eval_sol['score'] == added[1], \
            "Rank and score not as expected: Solution {}, expected added {}".format(eval_sol, added)


def test_tournament_selector_better_rank():
    sol = [{'rank': 0, 'score': 10},
           {'rank': 1, 'score': 10}]

    selector = parent_selection.get_tournament_selector()

    sel = selector(sol)

    assert sel is sol[0], "Wrong solution was chosen"


def test_tournament_selector_better_score_in_rank():
    sol = [{'rank': 0, 'score': 20},
           {'rank': 0, 'score': 10}]

    selector = parent_selection.get_tournament_selector()

    sel = selector(sol)

    assert sel is sol[0], "Wrong solution was chosen"

def test_tournament_selector_same():
    sol = [{'rank': 0, 'score': 10},
           {'rank': 1, 'score': 10}]

    selector = parent_selection.get_tournament_selector()

    np.random.seed(0)
    p = np.random.random()
    np.random.seed(0)
    sel = selector(sol)

    if p < 0.5:
        assert sel is sol[1], "Wrong solution was chosen"
    else:
        assert sel is sol[0], "Wrong solution was chosen"



def test_random_3point_crossover():
    cross_op = crossover.get_npoint_crossover(3)
    p1 = {'solution': np.ones(10, dtype=int)}
    p2 = {'solution': np.zeros(10, dtype=int)}

    c = cross_op(p1,p2)

    assert np.logical_or(c['solution'] == 1, c['solution'] == 0).all(), \
        "Crossover solution is not valid {}".format(c['solution'])


def test_4point_crossover():
    cross_op = crossover.get_npoint_crossover(4)

    crossover_points = np.array([0, 2, 5, 9])
    p1 = {'solution': np.ones(10, dtype=int)}
    p2 = {'solution': np.zeros(10, dtype=int)}

    expected = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])

    c = cross_op(p1, p2, crossover_points)

    assert (c['solution'] == expected).all(), \
    "Expected result from crossover is not real result: {}, {}".format(c['solution'], expected)


def test_10point_crossover():
    cross_op = crossover.get_npoint_crossover(10)

    crossover_points = np.array([0,1,2,3,4,5,6,7,8,9])
    p1 = {'solution': np.ones(10, dtype=int)}
    p2 = {'solution': np.zeros(10, dtype=int)}

    expected = np.array([0,1,0,1,0,1,0,1,0,1])
    c = cross_op(p1,p2,crossover_points)

    assert (c['solution'] == expected).all(), "Expected result from crossover is not real result"


def test_uniform_crossover():
    cross_op = crossover.get_uniform_crossover()

    p1 = {'solution': np.ones(10, dtype=int)}
    p2 = {'solution': np.zeros(10, dtype=int)}

    expected = np.array([1,0,1,0,1,0,1,0,1,0])

    c = cross_op(p1,p2)

    assert (c['solution'] == expected).all(), "Expected result from crossover is not real result {} {}" \
        .format(c['solution'], expected)


def test_enumeration():
    t1 = np.array([1,1,1])
    t2 = np.array([0,0,1])

    expected = [ {'solution': (0,1,1), 'evaluation': (1,1)},
                 {'solution': (1,1,1), 'evaluation': (0,2)},
                 {'solution': (1,0,1), 'evaluation': (1,1)},
                 {'solution': (0,0,1), 'evaluation': (2,0)},]

    objective = objectives.get_hamming_distance_objective(t1,t2)
    pareto_front = utils.pareto_by_enumeration(objective, product([1,0], repeat=3))

    assert len(pareto_front) == len(expected), "Pareto front has less solutions than expected"

    expected.sort(key = lambda solution: solution['evaluation'])
    pareto_front.sort(key = lambda solution: solution['evaluation'])

    for exp,real in zip(expected,pareto_front):
        assert exp['evaluation'] == real['evaluation'], "Solution in pareto front does not match real {}, {}".format(exp,real)


def test_NSGAII_ranker():

    eval_solutions = [{'evaluation': (1,4)},
                     {'evaluation': (4,0)},
                     {'evaluation': (2,2)},
                     {'evaluation': (3,1)},
                     {'evaluation': (4,3)},
                     {'evaluation': (3,4)}]


    eval_solutions_added = [(0,inf),
                          (0,inf),
                          (0,10),
                          (0,8),
                          (1,inf),
                          (1,inf)]

    ranker = ranking.get_NSGAII_ranker()

    out = ranker(eval_solutions)

    p = [out[i]['score'] >= out[i+1]['score'] if out[i]['rank'] == out[i+1]['rank']
         else out[i]['rank'] < out[i+1]['rank'] for i in range(len(out)-1)]

    assert all(p)

    for eval_sol, added in zip(eval_solutions, eval_solutions_added):
        assert eval_sol['rank'] == added[0] and eval_sol['score'] == added[1], \
            "Solution {}, added {}".format(eval_sol, added)


def test_NSGAII_ranker_2():
    eval_solutions = [{'evaluation': (1,4)},
                     {'evaluation': (4,0)},
                     {'evaluation': (2,2)},
                     {'evaluation': (3,1)},
                     {'evaluation': (4,3)}]

    eval_solutions_added = [(0,inf),
                            (0,inf),
                            (0,10),
                            (0,8),
                            (1,inf)]

    ranker = ranking.get_NSGAII_ranker()

    out = ranker(eval_solutions)

    p = [out[i]['score'] >= out[i+1]['score'] if out[i]['rank'] == out[i+1]['rank']
         else out[i]['rank'] < out[i+1]['rank'] for i in range(len(out)-1)]

    assert all(p)

    for eval_sol, added in zip(eval_solutions, eval_solutions_added):
        assert eval_sol['rank'] == added[0] and eval_sol['score'] == added[1], \
            "Solution {}, added {}".format(eval_sol, added)


def test_from_all_selection1():
    reference_point = (5,5)
    mu = 5
    parents = [{'solution': 3, 'evaluation': (2,2)},
               {'solution': 4, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (2,2)},
               {'solution': 3, 'evaluation': (2,2)}]


    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 4, 'evaluation': (3,4)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_all(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   2: 1,
                   3: 2,
                   4: 1 }

    for s in new_gen:
        occurences[s['solution']] -= 1


    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"

def test_from_all_selection2():
    reference_point = (5,5)
    mu = 4
    parents = [{'solution': 3, 'evaluation': (2,2)},
               {'solution': 4, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)},
               {'solution': 3, 'evaluation': (2,2)}]


    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 4, 'evaluation': (3,4)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_all(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   2: 1,
                   3: 2
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"


def test_from_all_selection3():
    reference_point = (5,5)
    mu = 3
    parents = [{'solution': 3, 'evaluation': (2,2)},
               {'solution': 4, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)},
               {'solution': 3, 'evaluation': (2,2)}]


    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 4, 'evaluation': (3,4)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_all(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   2: 1,
                   3: 1
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"


def test_from_all_selection4():
    reference_point = (5,5)
    mu = 2
    parents = [{'solution': 3, 'evaluation': (2,2)},
               {'solution': 4, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)},
               {'solution': 3, 'evaluation': (2,2)}]


    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 4, 'evaluation': (3,4)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_all(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   3: 1
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"


def test_from_offspring_selection1():
    reference_point = (5,5)
    mu = 3

    parents = [{'solution': 0, 'evaluation': (2,2)},
               {'solution': 0, 'evaluation': (2,2)}]

    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 3, 'evaluation': (3,4)},
               {'solution': 3, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_offspring(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   2: 1,
                   3: 1
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"


def test_from_offspring_selection2():
    reference_point = (5,5)
    mu = 2

    parents = [{'solution': 0, 'evaluation': (2,2)},
               {'solution': 0, 'evaluation': (2,2)}]

    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 3, 'evaluation': (3,4)},
               {'solution': 3, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_offspring(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                   2: 1
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"


def test_from_offspring_selection3():
    reference_point = (5,5)
    mu = 1

    parents = [{'solution': 0, 'evaluation': (2,2)},
               {'solution': 0, 'evaluation': (2,2)}]

    children = [{'solution': 2, 'evaluation': (1,4)},
               {'solution': 3, 'evaluation': (3,4)},
               {'solution': 3, 'evaluation': (4,3)},
               {'solution': 1, 'evaluation': (3,1)}]


    ranker = ranking.get_SMSEMOA_ranker(reference_point)
    selector = selection.get_from_offspring(ranker)

    new_gen = selector(parents,children, mu)

    assert len(new_gen) == mu, "Incorrect number of new solutions"

    occurences = { 1: 1,
                 }

    for s in new_gen:
        occurences[s['solution']] -= 1

    assert all([occurences[key] == 0 for key in occurences]), "Wrong solutions have been selected"



def test_binary_list_initializer():
    initializer = solution.get_random_binary_list_initializor(5)
    gen = initializer()['solution']

    assert gen.shape == (5,), "Solution of invalid shape generated {}".format(gen)

    assert np.logical_or(gen == 1, gen == 0).all(), "Invalid solution generated {}".format(gen)


def test_hyperspace_volume_dominated():
    rp = (7,1)

    sols = [{'evaluation': (0,0)},
             {'evaluation': (2,-2)},
             {'evaluation': (3,-4)},
             {'evaluation': (3,-4)},
             {'evaluation': (4,-5)},
             {'evaluation': (4,-5)},
             {'evaluation': (6,-7)},
             {'evaluation': (5,-1)},
             {'evaluation': (4,-3)},
             {'evaluation': (4,-3)}]


    dominated = utils.calculate_hyperspace_volume(rp,sols)

    assert dominated == 30, "Incorrect hyperspace volume: {} is not 30".format(dominated)

