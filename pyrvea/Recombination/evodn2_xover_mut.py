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

        r = min(len(sub1), len(sub2))

        for layer in range(r):

            connections = min(sub1[layer][1:, :].size, sub2[layer][1:, :].size)

            # Crossover
            exchange = sample(
                range(connections), np.random.binomial(connections, prob_crossover)
            )
            tmp = deepcopy(sub1[layer])
            sub1[layer][1:, :].ravel()[exchange] = sub2[layer][1:, :].ravel()[exchange]
            sub2[layer][1:, :].ravel()[exchange] = tmp[1:, :].ravel()[exchange]

            try:
                # Mutate first individual
                connections = min(
                    sub1[layer][1:, :].size,
                    sub2[layer][1:, :].size,
                    sub3[layer][1:, :].size,
                    sub4[layer][1:, :].size,
                )
                mutate = sample(
                    range(connections), np.random.binomial(connections, prob_mut)
                )
                sub1[layer][1:, :].ravel()[mutate] = sub1[layer][1:, :].ravel()[
                                                         mutate
                                                     ] + mut_strength * 0 * (1 - cur_gen / total_gen) * (
                                                             sub3[layer][1:, :].ravel()[mutate]
                                                             - sub4[layer][1:, :].ravel()[mutate]
                                                     )

                # Mutate second individual
                mutate = sample(
                    range(connections), np.random.binomial(connections, prob_mut)
                )
                sub2[layer][1:, :].ravel()[mutate] = sub2[layer][1:, :].ravel()[
                                                         mutate
                                                     ] + mut_strength * 0 * (1 - cur_gen / total_gen) * (
                                                             sub3[layer][1:, :].ravel()[mutate]
                                                             - sub4[layer][1:, :].ravel()[mutate]
                                                     )
            except IndexError:


                break

    return offspring1, offspring2
