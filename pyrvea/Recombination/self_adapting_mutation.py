from numpy import np
from random import sample


def mutation(
    population,
    offspring1,
    offspring2,
    cur_gen,
    total_gen,
    prob_mut=0.3,
    mut_strength=0.7,
):

    individuals = population.individuals

    o1 = individuals[offspring1]
    o2 = individuals[offspring2]

    # Could call the delete method from population_evonn class here
    indices = [offspring1, offspring2]
    mask = np.ones(len(individuals), dtype=bool)
    mask[indices] = False

    alternatives = individuals[mask, ...]

    it = np.nditer([o1, o2], flags=["multi_index"], op_flags=["readwrite"])
    for wx1, wx2 in it:
        if wx1 != 0 and np.random.random() < prob_mut:

            select = sample(list(alternatives), 2)

            o1[it.multi_index[0], it.multi_index[1]] = wx1 + mut_strength * (
                1 - cur_gen / total_gen
            ) * (
                select[0][it.multi_index[0], it.multi_index[1]]
                - select[1][it.multi_index[0], it.multi_index[1]]
            )

        if wx2 != 0 and np.random.random() < prob_mut:

            select = sample(list(alternatives), 2)

            o2[it.multi_index[0], it.multi_index[1]] = wx1 + mut_strength * (
                1 - cur_gen / total_gen
            ) * (
                select[0][it.multi_index[0], it.multi_index[1]]
                - select[1][it.multi_index[0], it.multi_index[1]]
            )

    return offspring1, offspring2
