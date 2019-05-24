import numpy as np
from random import sample


def mutation(
    population,
    offspring1_idx,
    offspring2_idx,
    cur_gen=1,
    total_gen=10,
    prob_mut=0.3,
    mut_strength=0.7,
):
    def mutate(offspring, wx):

        select = sample(list(alternatives), 2)

        offspring[it.multi_index[0], it.multi_index[1]] = wx + mut_strength * (
            1 - cur_gen / total_gen
        ) * (
            select[0][it.multi_index[0], it.multi_index[1]]
            - select[1][it.multi_index[0], it.multi_index[1]]
        )

        return offspring

    individuals = population.individuals

    o1 = individuals[offspring1_idx]
    o2 = individuals[offspring2_idx]

    # Could call the delete method from population_evonn class here
    indices = [offspring1_idx, offspring2_idx]
    mask = np.ones(len(individuals), dtype=bool)
    mask[indices] = False

    alternatives = individuals[mask, ...]

    it = np.nditer([o1, o2], flags=["multi_index"], op_flags=["readwrite"])

    for wx1, wx2 in it:
        if wx1 != 0 and np.random.random() < prob_mut:

            o1 = mutate(o1, wx1)

        if wx2 != 0 and np.random.random() < prob_mut:

            o2 = mutate(o2, wx2)

    return offspring1_idx, offspring2_idx
