import numpy as np
from random import shuffle


def mate(
    mating_pop, individuals: list, params, crossover_type=None, mutation_type=None
):
    """Swap nodes between two partners and mutate based on standard deviation.

    Parameters
    ----------
    mating_pop : list
        List of indices of individuals to mate. If None, choose from population
        randomly.
        Each entry should contain two indices, one for each parent.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for evolution. If None, use defaults.

    Returns
    -------
    offspring : list
        The offsprings produced as a result of crossover and mutation.
    """

    prob_crossover = params.get("prob_crossover", 0.8)
    prob_mutation = params.get("prob_mutation", 0.3)
    mut_strength = params.get("mut_strength", 1.0)
    cur_gen = params.get("current_total_gen_count", 1)
    total_gen = params.get("total_generations", 10)
    std_dev = (5 / 3) * (1 - cur_gen / total_gen)
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

        if mutation_type == "gaussian" or mutation_type is None:
            # Method : Gaussian (default)
            # Take a random number of connections based on probability and mutate based
            # on standard deviation, calculated once per generation.

            connections = offspring1.size

            mut_val = np.random.normal(0, std_dev, connections) * mut_strength

            mut = np.random.choice(
                connections,
                np.random.binomial(connections, prob_mutation),
                replace=False,
            )
            offspring1.ravel()[mut] += offspring1.ravel()[mut] * mut_val[mut]

            mut_val = np.random.normal(0, std_dev, connections) * mut_strength

            mut = np.random.choice(
                connections,
                np.random.binomial(connections, prob_mutation),
                replace=False,
            )
            offspring2.ravel()[mut] += offspring2.ravel()[mut] * mut_val[mut]

        elif mutation_type == "self-adapting":
            # Method: Self adapting mutation
            # Choose two random individuals and a random number of connections,
            # mutate offspring based on current gen and connections of two randomly
            # chosen individuals

            # Randomly select two individuals with current match active (=non-zero)
            connections = offspring1.size
            select = np.asarray(individuals)[
                np.random.choice(np.nonzero(np.asarray(individuals))[0], 2)
            ]

            mut = np.random.choice(
                connections,
                np.random.binomial(connections, prob_mutation),
                replace=False,
            )
            offspring1.ravel()[mut] = offspring1.ravel()[mut] + mut_strength * (
                1 - cur_gen / total_gen
            ) * (select[1].ravel()[mut] - select[0].ravel()[mut])

            select = np.asarray(individuals)[
                np.random.choice(np.nonzero(np.asarray(individuals))[0], 2)
            ]

            mut = np.random.choice(
                connections,
                np.random.binomial(connections, prob_mutation),
                replace=False,
            )
            offspring2.ravel()[mut] = offspring2.ravel()[mut] + mut_strength * (
                1 - cur_gen / total_gen
            ) * (select[1].ravel()[mut] - select[0].ravel()[mut])

        else:
            pass

        offspring.extend((offspring1, offspring2))

    return offspring


class EvoNNRecombination:
    def __init__(
        self,
        evolver: "BaseEA",
        ProC: float = 0.8,
        ProM: float = 0.3,
        mutation_strength: float = 1.0,
        mutation_type: str = "gaussian",
    ):
        self.evolver = evolver
        self.mutation_type: str = mutation_type
        self.ProC: float = ProC
        self.ProM: float = ProM
        self.mutation_strength: float = mutation_strength

    def do(self, pop, mating_pop_ids: list = None):
        cur_gen = self.evolver.__getattribute__("_current_gen_count")
        total_gen = self.evolver.__getattribute__("total_gen_count")
        pop_size = pop.shape[0]
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        # TODO fix the need for the following
        # [1,2,3,4] -> [[1,2],[3,4]]
        if np.asarray(shuffled_ids).ndim == 1:
            mating_pop = [
                [shuffled_ids[x], shuffled_ids[x]]
                for x in range(len(shuffled_ids))
                if x % 2 == 0
            ]
        elif np.asarray(shuffled_ids).ndim == 2:
            mating_pop = shuffled_ids
        std_dev = (5 / 3) * (1 - cur_gen / total_gen)
        offspring = []
        for mates in mating_pop:

            offspring1 = np.copy(pop[mates[0]])
            offspring2 = np.copy(pop[mates[1]])

            # Crossover
            for i in range(offspring1.shape[1]):
                if np.random.random() < self.ProC:
                    tmp = np.copy(offspring1[:, i])
                    offspring1[:, i] = offspring2[:, i]
                    offspring2[:, i] = tmp

            if self.mutation_type == "gaussian" or self.mutation_type is None:
                # Method : Gaussian (default)
                # Take a random number of connections based on probability and mutate based
                # on standard deviation, calculated once per generation.

                connections = offspring1.size

                mut_val = (
                    np.random.normal(0, std_dev, connections) * self.mutation_strength
                )

                mut = np.random.choice(
                    connections,
                    np.random.binomial(connections, self.ProM),
                    replace=False,
                )
                offspring1.ravel()[mut] += offspring1.ravel()[mut] * mut_val[mut]

                mut_val = (
                    np.random.normal(0, std_dev, connections) * self.mutation_strength
                )

                mut = np.random.choice(
                    connections,
                    np.random.binomial(connections, self.ProM),
                    replace=False,
                )
                offspring2.ravel()[mut] += offspring2.ravel()[mut] * mut_val[mut]

            elif self.mutation_type == "self-adapting":
                # Method: Self adapting mutation
                # Choose two random individuals and a random number of connections,
                # mutate offspring based on current gen and connections of two randomly
                # chosen individuals

                # Randomly select two individuals with current match active (=non-zero)
                connections = offspring1.size
                select = np.asarray(pop)[
                    np.random.choice(np.nonzero(np.asarray(pop))[0], 2)
                ]

                mut = np.random.choice(
                    connections,
                    np.random.binomial(connections, self.ProM),
                    replace=False,
                )
                offspring1.ravel()[mut] = offspring1.ravel()[
                    mut
                ] + self.mutation_strength * (1 - cur_gen / total_gen) * (
                    select[1].ravel()[mut] - select[0].ravel()[mut]
                )

                select = np.asarray(pop)[
                    np.random.choice(np.nonzero(np.asarray(pop))[0], 2)
                ]

                mut = np.random.choice(
                    connections,
                    np.random.binomial(connections, self.ProM),
                    replace=False,
                )
                offspring2.ravel()[mut] = offspring2.ravel()[
                    mut
                ] + self.mutation_strength * (1 - cur_gen / total_gen) * (
                    select[1].ravel()[mut] - select[0].ravel()[mut]
                )

            else:
                pass

            offspring.extend((offspring1, offspring2))

        return np.asarray(offspring)
