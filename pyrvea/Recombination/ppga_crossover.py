import numpy as np


def mate(mating_pop, individuals, params):
    """Randomly exchange nodes between two individuals.

    Parameters
    ----------
    w1 : np.ndarray
        The first individual
    w2 : np.ndarray
        The second individual
    prob_crossover : float
        The probability for the crossover
    """

    prob_crossover = params["prob_crossover"]
    offspring1 = np.copy(individuals[mating_pop[0]])
    offspring2 = np.copy(individuals[mating_pop[1]])

    # Crossover

    # Method 1
    # Take a random number of connections based on probability and swap them
    #
    # connections = offspring1[1:, :].size
    # exchange = np.random.choice(connections, np.random.binomial(connections, prob_crossover), replace=False)
    # tmp = np.copy(offspring1)
    # offspring1[1:, :].ravel()[exchange] = offspring2[1:, :].ravel()[exchange]
    # offspring2[1:, :].ravel()[exchange] = tmp[1:, :].ravel()[exchange]

    # Method 2
    # Take random nodes based on probability and swap them
    #
    for i in range(offspring1.shape[1]):
        if np.random.random() < prob_crossover:
            tmp = np.copy(offspring1[:, i])
            offspring1[:, i] = offspring2[:, i]
            offspring2[:, i] = tmp

    offspring = [offspring1, offspring2]
    return offspring
