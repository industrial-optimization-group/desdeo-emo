import numpy as np
from copy import deepcopy
from random import sample
from math import ceil
from timeit import default_timer as timer


def mutate(
    mating_pop,
    individuals,
    params,
    lower_limits=None,
    upper_limits=None
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
    prob_mutation : float
        The probability for mutation
    mut_strength : float
        Mutation alfa parameter
    cur_gen : int
        Current generation
    total_gen : int
        Total generations
    std_dev : float
        Standard deviation
    """

    parent1, parent2 = mating_pop[0], mating_pop[1]
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)

    cur_gen = params["current_total_gen_count"]
    total_gen = params["total_generations"]
    prob_crossover = params["prob_crossover"]
    prob_mutation = params["prob_mutation"]
    std_dev = params["std_dev"]

    connections = offspring1[1:, :].size

    # Method 1: Gaussian
    # Take a random number of connections based on probability and mutate based on
    # standard deviation, calculated once per generation
    # VERY FAST
    #
    mut_val = np.random.normal(0, std_dev, connections)

    mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
    offspring1[1:, :].ravel()[mut] += offspring1[1:, :].ravel()[mut] * mut_val[mut]

    mut_val = np.random.normal(0, std_dev, connections)

    mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
    offspring2[1:, :].ravel()[mut] += offspring2[1:, :].ravel()[mut] * mut_val[mut]

    # Method 2
    # Choose two random individuals and a random number of connections,
    # mutate offspring based on current gen and connections of two randomly chosen individuals
    #
    # alternatives = individuals[:, 1:, :]
    #
    # # Randomly select two individuals with current match active (=non-zero)
    # select = alternatives[
    #     np.random.choice(
    #         np.nonzero(alternatives)[
    #             0
    #         ],
    #         2,
    #     )
    # ]
    #
    # mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
    # offspring1[1:, :].ravel()[mut] = offspring1[1:, :].ravel()[mut] + params["mut_strength"] * (
    #             1 - cur_gen / total_gen
    #         ) * (select[1].ravel()[mut] - select[0].ravel()[mut])
    #
    # select = alternatives[
    #     np.random.choice(
    #         np.nonzero(alternatives)[
    #             0
    #         ],
    #         2,
    #     )
    # ]
    #
    # mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
    # offspring2[1:, :].ravel()[mut] = offspring2[1:, :].ravel()[mut] + params["mut_strength"] * (
    #             1 - cur_gen / total_gen
    #         ) * (select[1].ravel()[mut] - select[0].ravel()[mut])