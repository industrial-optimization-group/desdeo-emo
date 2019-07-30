from pyrvea.Selection.tournament_select import tour_select
import numpy as np
from copy import deepcopy


def height_fair_xover(offspring1, offspring2):

    rand_subtree = np.random.randint(min(len(offspring1), len(offspring2)))

    if len(offspring1[rand_subtree].roots) == 0 or len(offspring2[rand_subtree].roots) == 0:
        tmp = deepcopy(offspring1[rand_subtree])
        offspring1[rand_subtree] = offspring2[rand_subtree]
        offspring2[rand_subtree] = tmp

    else:
        height_fair_xover(offspring1[rand_subtree].roots, offspring2[rand_subtree].roots)


def mate(mating_pop, individuals, params):

    if mating_pop is None:
        mating_pop = []
        for i in range(int(len(individuals) / 2)):
            mating_pop.append(
                [
                    tour_select(individuals, params["tournament_size"]),
                    tour_select(individuals, params["tournament_size"]),
                ]
            )

    offspring = []

    for mates in mating_pop:

        offspring1 = deepcopy(individuals[mates[0]])
        offspring2 = deepcopy(individuals[mates[1]])

        if np.random.rand() < 0.5:
            rand_subtree = np.random.randint(len(offspring1.roots))
            tmp = deepcopy(offspring1.roots[rand_subtree])
            offspring1.roots[rand_subtree] = offspring2.roots[rand_subtree]
            offspring2.roots[rand_subtree] = tmp

        else:
            height_fair_xover(offspring1.roots, offspring2.roots)

        offspring.extend((offspring1, offspring2))

    return offspring
