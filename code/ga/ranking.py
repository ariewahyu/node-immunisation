from math import inf
from . import utils

def get_ranking_groups(evaluated_solutions):
    to_rank = evaluated_solutions[:]
    ranked = []

    while to_rank:
        dominated = []
        this_rank = []

        to_rank.sort(key=lambda solution: solution['evaluation'])

        to_rank[0]['rank'] = len(ranked)
        this_rank.append(to_rank[0])
        i = 0
        j = 1
        while j < len(to_rank):
            if not utils.is_dominated_by(to_rank[j]['evaluation'], to_rank[i]['evaluation']):
                to_rank[j]['rank'] = len(ranked)
                this_rank.append(to_rank[j])
                i = j
                j = i+1
            else:
                dominated.append(to_rank[j])
                j += 1

        ranked.append(this_rank)
        to_rank = dominated

    return ranked


def get_hyperspace_contributions_and_sort(ranking_groups, reference_point):
    for group in ranking_groups:

        for i, sol in enumerate(group):
            if i == 0:
                point1 = (reference_point[1] - sol['evaluation'][1])
            else:
                point1 = (group[i-1]['evaluation'][1] - sol['evaluation'][1])


            if i == len(group)-1:
                point2 = (reference_point[0] - sol['evaluation'][0])
            else:
                point2 = (group[i+1]['evaluation'][0] - sol['evaluation'][0])

            sol['score'] = point1*point2

        group.sort(key=lambda solution: solution['score'], reverse=True)


def get_crowding_distances_and_sort(ranking_groups):
    for group in ranking_groups:

        for i, sol in enumerate(group):
            if i in (0, len(group)-1):
                sol['score'] = inf
            else:
                sol['score'] = (2*(group[i+1]['evaluation'][0] - group[i-1]['evaluation'][0]) +
                                2*(group[i-1]['evaluation'][1] - group[i+1]['evaluation'][1]))

        group.sort(key=lambda solution: solution['score'], reverse=True)


def get_SMSEMOA_ranker(reference_point):

    def SMSEMOA_ranker(solutions):
        ranking_groups = get_ranking_groups(solutions)
        get_hyperspace_contributions_and_sort(ranking_groups, reference_point)

        flattened = [solution for group in ranking_groups for solution in group]
        return flattened

    return SMSEMOA_ranker


def get_NSGAII_ranker():

    def NSGAII_ranker(solutions):
        ranking_groups = get_ranking_groups(solutions)
        get_crowding_distances_and_sort(ranking_groups)

        flattened = [solution for group in ranking_groups for solution in group]
        return flattened

    return NSGAII_ranker
