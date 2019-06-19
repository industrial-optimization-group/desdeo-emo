import numpy as np
from copy import deepcopy
from random import sample


def ppga_parallel_crossover(parent1, parent2, prob_crossover=0.8):
    """Randomly exchange connections between two individuals.

    Parameters
    ----------
    parent1 : np.ndarray
        The first individual
    parent2 : np.ndarray
        The second individual
    prob_crossover : float
        The probability for the crossover
    """

    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)

    for subnet in range(len(offspring1)):

        sub1 = offspring1[subnet]
        sub2 = offspring2[subnet]

        if len(sub1) <= len(sub2):
            r = len(sub1)
        else:
            r = len(sub2)

        for layer in range(r):

            if sub1[layer].size <= sub2[layer].size:
                connections = sub1[layer][1:, :].size
            else:
                connections = sub2[layer][1:, :].size

            exchange = sample(range(connections), np.random.binomial(connections, prob_crossover))
            tmp = deepcopy(sub1[layer])
            sub1[layer][1:, :].ravel()[exchange] = sub2[layer][1:, :].ravel()[exchange]
            sub2[layer][1:, :].ravel()[exchange] = tmp[1:, :].ravel()[exchange]

    return offspring1, offspring2
