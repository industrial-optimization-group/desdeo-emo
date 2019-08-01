from pyrvea.Selection.tournament_select import tour_select
import numpy as np
from copy import deepcopy
from random import choice


def mate(mating_pop, individuals, params):

    prob_standard = 0.5
    prob_height_fair = params["prob_crossover"] - prob_standard

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

        # Height-fair xover
        if np.random.rand() <= prob_height_fair:
            depth = min(offspring1.max_depth, offspring2.max_depth)
            rand_node1 = choice(offspring1.nodes_at_depth[depth])
            rand_node2 = choice(offspring2.nodes_at_depth[depth])
            tmp_value = rand_node1.value
            tmp_roots = rand_node1.roots
            rand_node1.value = rand_node2.value
            rand_node1.roots = rand_node2.roots
            rand_node2.value = tmp_value
            rand_node2.roots = tmp_roots

        # Standard xover
        elif np.random.rand() <= prob_height_fair + prob_standard:
            rand_node1 = choice(offspring1.nodes[1:])  # Exclude linear node
            rand_node2 = choice(offspring2.nodes[1:])
            tmp_value = rand_node1.value
            tmp_roots = rand_node1.roots
            rand_node1.value = rand_node2.value
            rand_node1.roots = rand_node2.roots
            rand_node2.value = tmp_value
            rand_node2.roots = tmp_roots

        else:
            continue

        offspring.extend((offspring1, offspring2))

    return offspring
