import numpy as np
from copy import deepcopy
from random import choice


def mate(mating_pop, individuals, params):

    prob_standard = 0.5
    prob_height_fair = params["prob_crossover"] - prob_standard
    r = np.random.rand()

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []

    for mates in mating_pop:

        offspring1 = deepcopy(individuals[mates[0]])
        offspring2 = deepcopy(individuals[mates[1]])

        # Height-fair xover
        if r <= prob_height_fair:
            depth = min(offspring1.total_depth, offspring2.total_depth)
            rand_node1 = choice(offspring1.nodes_at_depth[depth])
            rand_node2 = choice(offspring2.nodes_at_depth[depth])
            tmp_value = rand_node1.value
            tmp_roots = rand_node1.roots
            rand_node1.value = rand_node2.value
            rand_node1.roots = rand_node2.roots
            rand_node2.value = tmp_value
            rand_node2.roots = tmp_roots
            offspring1.nodes = offspring1.get_sub_nodes()
            offspring2.nodes = offspring2.get_sub_nodes()

        # Standard xover
        elif r <= prob_height_fair + prob_standard:
            rand_node1 = choice(offspring1.nodes[1:])  # Exclude linear node
            rand_node2 = choice(offspring2.nodes[1:])
            tmp_value = rand_node1.value
            tmp_roots = rand_node1.roots
            rand_node1.value = rand_node2.value
            rand_node1.roots = rand_node2.roots
            rand_node2.value = tmp_value
            rand_node2.roots = tmp_roots
            offspring1.nodes = offspring1.get_sub_nodes()
            offspring2.nodes = offspring2.get_sub_nodes()

        else:
            pass

        offspring.extend((offspring1, offspring2))

    return offspring
