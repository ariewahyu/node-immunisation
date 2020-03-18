def get_from_all(ranker):

    def from_all(parents, children, select):
        new_gen = parents + children

        assert len(new_gen) >= select

        while len(new_gen) > select:
            ranking = ranker(new_gen)
            new_gen = ranking[:-1]

        return new_gen

    return from_all

def get_from_offspring(ranker):

    def from_offspring(_, children, select):
        new_gen = children[:]

        assert len(new_gen) >= select

        while len(new_gen) > select:
            ranking = ranker(new_gen)
            new_gen = ranking[:-1]

        return new_gen

    return from_offspring
