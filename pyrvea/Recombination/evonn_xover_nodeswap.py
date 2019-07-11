import numpy as np


def mate(mating_pop, individuals, params):
    """Randomly exchange nodes in the hidden layer between two parents and produce
    two offsprings.

    Parameters
    ----------
    mating_pop : list
        List of individuals to mate. If None, choose random partners from population.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for breeding. If None, use defaults.

    Returns
    -------
    offspring : list
        Two offsprings as a result of crossover.

    """

    try:
        prob_crossover = params["prob_crossover"]
    except KeyError:
        prob_crossover = 0.8

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []

    for mates in mating_pop:

        offspring1 = np.copy(individuals[mates[0]])
        offspring2 = np.copy(individuals[mates[1]])

        # Take random nodes on the hidden layer based on probability and swap them

        for i in range(offspring1.shape[1]):
            if np.random.random() < prob_crossover:
                tmp = np.copy(offspring1[:, i])
                offspring1[:, i] = offspring2[:, i]
                offspring2[:, i] = tmp

        offspring.extend((offspring1, offspring2))

    return offspring
