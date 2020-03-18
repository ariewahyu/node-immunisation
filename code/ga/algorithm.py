def genetic_algorithm(lda,
                      evaluations,
                      solution_initializer,
                      objective_function,
                      mutator,
                      selector,
                      parent_selector,
                      crossover=None,
                      after_gen=None):

    population = solution_initializer()
    mu = len(population)
    print(mu)

    for solution in population:
        objective_function(solution)

    for generation in range(0, evaluations):
        if generation % 100 == 0:
            print("Generation {}".format(generation))
            if(after_gen):
               after_gen(population)


        children = []
        for _ in range(0, lda):
            if crossover:
                parent1 = parent_selector(population)
                parent2 = parent_selector(population)
                child = crossover(parent1, parent2)
                mutated_child = mutator(child)
                objective_function(mutated_child)
                children.append(mutated_child)

            else:
                parent = parent_selector(population)
                child = mutator(parent)
                objective_function(child)
                children.append(child)

        population = selector(population, children, mu)

    return population
