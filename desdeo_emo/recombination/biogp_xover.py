import numpy as np
from copy import deepcopy
from random import choice
from random import shuffle


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
        List of indices of individuals to mate. If None, choose from population
        randomly.
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

    prob_crossover = params.get("prob_crossover", 0.9)

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


class BioGP_xover:
    def __init__(
        self, probability_crossover: float = 0.9, probability_standard: float = 0.5
    ):
        self.probability_crossover: float = probability_crossover
        if probability_standard > probability_crossover:
            msg = (
                f"Probability of standard crossover ({probability_standard}) should be "
                f"less than probability of crossover ({probability_crossover})"
            )
            raise ValueError(msg)
        self.probability_standard: float = probability_standard

    def do(self, pop, mating_pop_ids):
        prob_height_fair = self.probability_crossover - self.probability_standard
        pop_size = pop.shape[0]
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        if np.asarray(shuffled_ids).ndim == 1:
            mating_pop = [
                [shuffled_ids[x], shuffled_ids[x]]
                for x in range(len(shuffled_ids))
                if x % 2 == 0
            ]
        elif np.asarray(shuffled_ids).ndim == 2:
            mating_pop = shuffled_ids
        offspring = []
        r = np.random.rand()
        for mates in mating_pop:

            offspring1 = deepcopy(pop[mates[0]][0])
            offspring2 = deepcopy(pop[mates[1]][0])

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
            elif r <= self.probability_crossover:
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

        return np.asarray(offspring).reshape(-1,1)
