import numpy as np

def get_uniform_mutator(p):

    def uniform_mutator(s):
        mutated = np.array(s['solution'])

        selection = np.random.random(mutated.shape[0]) < p

        mutated[selection] = np.logical_not(mutated[selection])

        return {'solution': mutated}

    return uniform_mutator
