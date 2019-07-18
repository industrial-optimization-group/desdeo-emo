import numpy as np
from copy import deepcopy
from random import sample
from math import ceil
from timeit import default_timer as timer


def mate(mating_pop, individuals, params):
    """Swap nodes between two partners and mutate based on standard deviation.

    Parameters
    ----------
    mating_pop : list
        List of individuals to mate. If None, choose from population randomly.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for evolution. If None, use defaults.

    Returns
    -------
    offspring : list
        The offsprings produced as a result of crossover and mutation.
    """

    try:
        prob_crossover = params["prob_crossover"]
        prob_mutation = params["prob_mutation"]
        std_dev = params["std_dev"]
    except KeyError:
        prob_crossover = 0.8
        prob_mutation = 0.3
        std_dev = (5 / 3) * (
            1 - params["current_total_gen_count"] / params["total_generations"]
        )
        if std_dev < 0:
            std_dev = 0

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []

    for mates in mating_pop:

        offspring1, offspring2 = (
            deepcopy(individuals[mates[0]]),
            deepcopy(individuals[mates[1]]),
        )

        for subnet in range(len(offspring1)):

            sub1 = offspring1[subnet]
            sub2 = offspring2[subnet]

            for layer in range(max(len(sub1), len(sub2))):

                try:
                    connections = min(sub1[layer].size, sub2[layer].size)

                    # Crossover
                    exchange = np.random.choice(
                        connections,
                        np.random.binomial(connections, prob_crossover),
                        replace=False,
                    )
                    tmp = np.copy(sub1[layer])
                    sub1[layer].ravel()[exchange] = sub2[layer].ravel()[
                        exchange
                    ]
                    sub2[layer].ravel()[exchange] = tmp.ravel()[exchange]

                except IndexError:
                    pass

                # Mutate the first offspring
                try:
                    connections = sub1[layer].size

                    mut_val = np.random.normal(0, std_dev, connections)

                    mut = np.random.choice(
                        connections,
                        np.random.binomial(connections, prob_mutation),
                        replace=False,
                    )
                    sub1[layer].ravel()[mut] += (
                        sub1[layer].ravel()[mut] * mut_val[mut]
                    )

                except IndexError:
                    pass

                # Mutate the second offspring
                try:
                    connections = sub2[layer].size

                    mut_val = np.random.normal(0, std_dev, connections)

                    mut = np.random.choice(
                        connections,
                        np.random.binomial(connections, prob_mutation),
                        replace=False,
                    )
                    sub2[layer].ravel()[mut] += (
                        sub2[layer].ravel()[mut] * mut_val[mut]
                    )

                except IndexError:
                    continue

        offspring.extend((offspring1, offspring2))

    return offspring
