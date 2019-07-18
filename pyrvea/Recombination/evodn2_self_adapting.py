import numpy as np
from copy import deepcopy
from random import sample


def mate(mating_pop, individuals, params):
    """Swap nodes between two partners and do self-adapting mutation.

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
        prob_mut = params["prob_mutation"]
        mut_strength = params["mut_strength"]
        cur_gen = params["current_total_gen_count"]
        total_gen = params["total_generations"]

    except KeyError:
        prob_crossover = 0.8
        prob_mut = 0.3
        mut_strength = 0.7
        cur_gen = 1
        total_gen = 10

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
            sub3 = individuals[np.random.randint(0, len(individuals) - 1)][subnet]
            sub4 = individuals[np.random.randint(0, len(individuals) - 1)][subnet]

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

                try:
                    # Mutate first individual
                    connections = min(
                        sub1[layer].size,
                        sub3[layer].size,
                        sub4[layer].size,
                    )
                    mutate = sample(
                        range(connections), np.random.binomial(connections, prob_mut)
                    )
                    sub1[layer].ravel()[mutate] = sub1[layer].ravel()[
                        mutate
                    ] + mut_strength * (1 - cur_gen / total_gen) * (
                        sub3[layer].ravel()[mutate]
                        - sub4[layer].ravel()[mutate]
                    )

                    # Mutate second individual
                    connections = min(
                        sub2[layer].size,
                        sub3[layer].size,
                        sub4[layer].size,
                    )
                    mutate = sample(
                        range(connections), np.random.binomial(connections, prob_mut)
                    )
                    sub2[layer].ravel()[mutate] = sub2[layer].ravel()[
                        mutate
                    ] + mut_strength * (1 - cur_gen / total_gen) * (
                        sub3[layer].ravel()[mutate]
                        - sub4[layer].ravel()[mutate]
                    )
                except IndexError:

                    break
        offspring.extend((offspring1, offspring2))

    return offspring
