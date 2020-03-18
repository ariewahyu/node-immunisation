import numpy as np

def get_tournament_selector():
    def tournament_selector(solutions):

        assert len(solutions) > 1, "Tournament selection needs at least a population size of 2"

        participants = np.random.choice(len(solutions), size=2, replace=False)
        s1 = solutions[participants[0]]
        s2 = solutions[participants[1]]

        if s1['rank'] < s2['rank']:
            return s1
        if s1['rank'] > s2['rank']:
            return s2
        if s1['score'] > s2['score']:
            return s1
        if s1['score'] < s2['score']:
            return s2

        if np.random.random() < 0.5:
            return s1

        return s2

    return tournament_selector


def get_random_selector():

    def random_selector(solutions):
        return solutions[np.random.randint(len(solutions))]

    return random_selector
