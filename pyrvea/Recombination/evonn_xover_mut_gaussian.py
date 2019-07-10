import numpy as np
from random import shuffle


def mate(mating_pop, individuals, params):
    """Randomly exchange nodes between two individuals.

    Parameters
    ----------
    mating_pop : list
        List of individuals to mate
    individuals : list
        List of all individuals
    params : dict
        Parameters for evolution. If None, use defaults.
    """

    try:
        prob_crossover = params["prob_crossover"]
        prob_mutation = params["prob_mutation"]
        std_dev = params["std_dev"]
    except KeyError:
        prob_crossover = 0.8
        prob_mutation = 0.3
        std_dev = (4 / 3) * (
            1
            - params["current_total_gen_count"] / params["total_generations"]
        )
        if std_dev < 0:
            std_dev = 0

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []

    for mates in mating_pop:

        offspring1 = np.copy(individuals[mates[0]])
        offspring2 = np.copy(individuals[mates[1]])

        # Crossover
        for i in range(offspring1.shape[1]):
            if np.random.random() < prob_crossover:
                tmp = np.copy(offspring1[:, i])
                offspring1[:, i] = offspring2[:, i]
                offspring2[:, i] = tmp

        # Mutation

        connections = offspring1[1:, :].size

        mut_val = np.random.normal(0, std_dev, connections)

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        offspring1[1:, :].ravel()[mut] += offspring1[1:, :].ravel()[mut] * mut_val[mut]

        mut_val = np.random.normal(0, std_dev, connections)

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        offspring2[1:, :].ravel()[mut] += offspring2[1:, :].ravel()[mut] * mut_val[mut]

        offspring.extend((offspring1, offspring2))

    return offspring
