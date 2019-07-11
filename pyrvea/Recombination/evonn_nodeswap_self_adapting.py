import numpy as np


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
        cur_gen = params["current_total_gen_count"]
        total_gen = params["total_generations"]
        prob_crossover = params["prob_crossover"]
        prob_mutation = params["prob_mutation"]
        mut_strength = params["mut_strength"]

    except KeyError:
        cur_gen = 1
        total_gen = 10
        prob_crossover = 0.8
        prob_mutation = 0.3
        mut_strength = 0.7

    if mating_pop is None:
        mating_pop = []
        for i in range(len(individuals)):
            mating_pop.append([i, np.random.randint(len(individuals))])

    offspring = []
    alternatives = np.array(individuals)[:, 1:, :]

    for mates in mating_pop:

        offspring1 = np.copy(individuals[mates[0]])
        offspring2 = np.copy(individuals[mates[1]])

        # Crossover
        for i in range(offspring1.shape[1]):
            if np.random.random() < prob_crossover:
                tmp = np.copy(offspring1[:, i])
                offspring1[:, i] = offspring2[:, i]
                offspring2[:, i] = tmp

        # Method: Self adapting mutation
        # Choose two random individuals and a random number of connections,
        # mutate offspring based on current gen and connections of two randomly chosen individuals

        # Randomly select two individuals with current match active (=non-zero)
        connections = offspring1[1:, :].size
        select = alternatives[
            np.random.choice(
                np.nonzero(alternatives)[
                    0
                ],
                2,
            )
        ]

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        offspring1[1:, :].ravel()[mut] = offspring1[1:, :].ravel()[mut] + mut_strength * (
                    1 - cur_gen / total_gen
                ) * (select[1].ravel()[mut] - select[0].ravel()[mut])

        select = alternatives[
            np.random.choice(
                np.nonzero(alternatives)[
                    0
                ],
                2,
            )
        ]

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        offspring2[1:, :].ravel()[mut] = offspring2[1:, :].ravel()[mut] + mut_strength * (
                    1 - cur_gen / total_gen
                ) * (select[1].ravel()[mut] - select[0].ravel()[mut])

        offspring.extend((offspring1, offspring2))

    return offspring
