import numpy as np
from copy import deepcopy


def mate(mating_pop, individuals, params):

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []

    for mates in mating_pop:

        offspring1 = deepcopy(individuals[mates[0]])
        offspring2 = deepcopy(individuals[mates[1]])

        rand_subtree = np.random.randint(len(offspring1.roots))
        tmp = deepcopy(offspring1.roots[rand_subtree])
        offspring1.roots[rand_subtree] = offspring2.roots[rand_subtree]
        offspring2.roots[rand_subtree] = tmp

        offspring.extend((offspring1, offspring2))

    return offspring
