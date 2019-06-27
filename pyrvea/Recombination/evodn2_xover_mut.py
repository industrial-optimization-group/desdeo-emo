import numpy as np
from copy import deepcopy
from random import sample


def evodn2_xover_mut(
    parent1,
    parent2,
    individuals,
    prob_crossover=0.8,
    prob_mut=0.3,
    mut_strength=0.7,
    cur_gen=1,
    total_gen=10,
):
    """ Perform simultaneous crossover and mutation over two individuals.

    Parameters
    ----------
    parent1 : ndarray
        The first individual
    parent2 : ndarray
        The second individual
    individuals : ndarray
        All individuals to choose mutation partner from
    prob_crossover : float
        The probability for crossover
    prob_mut : float
        The probability for mutation
    mut_strength : float
        Mutation alfa parameter
    cur_gen : int
        Current generation
    total_gen : int
        Total generations
    """

    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)

    for subnet in range(len(offspring1)):

        sub1 = offspring1[subnet]
        sub2 = offspring2[subnet]
        sub3 = individuals[np.random.randint(0, individuals.shape[0] - 1)][subnet]
        sub4 = individuals[np.random.randint(0, individuals.shape[0] - 1)][subnet]

        r = max(len(sub1), len(sub2))

        for layer in range(r):

            try:
                connections = min(sub1[layer][1:, :].size, sub2[layer][1:, :].size)

                # Crossover
                exchange = np.random.choice(connections, np.random.binomial(connections, prob_crossover), replace=False)

                tmp = deepcopy(sub1[layer])
                sub1[layer][1:, :].ravel()[exchange] = sub2[layer][1:, :].ravel()[exchange]
                sub2[layer][1:, :].ravel()[exchange] = tmp[1:, :].ravel()[exchange]

            except IndexError:
                pass

            try:
                # Mutate first individual
                connections = sub1[layer][1:, :].size
                mutate = np.random.choice(connections, np.random.binomial(connections, prob_mut), replace=False)

                sub1[layer][1:, :].ravel()[mutate] = sub1[layer][1:, :].ravel()[
                    mutate
                ] + mut_strength * (1 - cur_gen / total_gen) * (
                    np.copy(sub3[layer][1:, :].ravel()[mutate])
                    - np.copy(sub4[layer][1:, :].ravel()[mutate])
                )
            except IndexError:
                pass

            try:
                # Mutate second individual
                connections = sub2[layer][1:, :].size
                mutate = np.random.choice(connections, np.random.binomial(connections, prob_mut), replace=False)

                sub2[layer][1:, :].ravel()[mutate] = sub2[layer][1:, :].ravel()[
                    mutate
                ] + mut_strength * (1 - cur_gen / total_gen) * (
                    np.copy(sub3[layer][1:, :].ravel()[mutate])
                    - np.copy(sub4[layer][1:, :].ravel()[mutate])
                )
            except IndexError:
                continue

    return offspring1, offspring2
