import numpy as np
from copy import deepcopy
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

    # Gaussian
    if mutation_type == "gaussian" or mutation_type is None:

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
                        sub1[layer].ravel()[exchange] = sub2[layer].ravel()[exchange]
                        sub2[layer].ravel()[exchange] = tmp.ravel()[exchange]

                    except IndexError:
                        pass

                    # Mutate the first offspring
                    try:
                        connections = sub1[layer].size

                        mut_val = (
                            np.random.normal(0, std_dev, connections) * mut_strength
                        )

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

                        mut_val = (
                            np.random.normal(0, std_dev, connections) * mut_strength
                        )

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

    else:
        pass

    return offspring


class EvoDN2Recombination:
    def __init__(
        self,
        evolver: "BaseEA",
        ProC: float = 0.8,
        ProM: float = 0.3,
        mutation_strength: float = 1.0,
    ):
        self.evolver = evolver
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
        if std_dev < 0:
            std_dev = 0

        offspring = []

        for mates in mating_pop:

            offspring1, offspring2 = (deepcopy(pop[mates[0]]), deepcopy(pop[mates[1]]))

            for subnet in range(len(offspring1)):

                sub1 = offspring1[subnet]
                sub2 = offspring2[subnet]

                for layer in range(max(len(sub1), len(sub2))):

                    try:
                        connections = min(sub1[layer].size, sub2[layer].size)

                        # Crossover
                        exchange = np.random.choice(
                            connections,
                            np.random.binomial(connections, self.ProC),
                            replace=False,
                        )
                        tmp = np.copy(sub1[layer])
                        sub1[layer].ravel()[exchange] = sub2[layer].ravel()[exchange]
                        sub2[layer].ravel()[exchange] = tmp.ravel()[exchange]

                    except IndexError:
                        pass

                    # Mutate the first offspring
                    try:
                        connections = sub1[layer].size

                        mut_val = (
                            np.random.normal(0, std_dev, connections)
                            * self.mutation_strength
                        )

                        mut = np.random.choice(
                            connections,
                            np.random.binomial(connections, self.ProM),
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

                        mut_val = (
                            np.random.normal(0, std_dev, connections)
                            * self.mutation_strength
                        )

                        mut = np.random.choice(
                            connections,
                            np.random.binomial(connections, self.ProM),
                            replace=False,
                        )
                        sub2[layer].ravel()[mut] += (
                            sub2[layer].ravel()[mut] * mut_val[mut]
                        )

                    except IndexError:
                        continue

            offspring.extend((offspring1, offspring2))
        return np.asarray(offspring)
