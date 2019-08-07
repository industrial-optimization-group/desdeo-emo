import numpy as np
from copy import deepcopy
from random import choice


def mate(mating_pop, individuals: list, params):
    """Perform BioGP crossover functions. Produce two offsprings by swapping genetic
    material of the two parents.

    Standard crossover:
    Swap two random subtrees between the parents.

    Height-fair crossover:
    Swap two random subtrees between the parents at the selected depth.

    Parameters
    ----------
    mating_pop : list
        List of indices of individuals to mate. If None, choose from population randomly.
        Each entry should contain two indices, one for each parent.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for evolution. If None, use defaults.

    Returns
    -------
    offspring : list
        The offsprings produced as a result of crossover.
    """

    prob_crossover = params.get("prob_crossover", 1.0)

    prob_standard = 0.5
    prob_height_fair = prob_crossover - prob_standard
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
