import numpy as np
import random

# Create w1 and w2 for for testing
individuals = np.random.uniform(-5, 5, size=(50, 6, 4))
w1 = random.randint(0, 50 - 1)
w2 = random.randint(0, 50 - 1)


def crossover(w1, w2, prob_xover=0.8):

    for i in range(np.shape(individuals[w1])[1]):
        if random.random() < prob_xover:
            tmp = np.copy(individuals[w1][:, i])
            individuals[w1][:, i] = individuals[w2][:, i]
            individuals[w2][:, i] = tmp

    return w1, w2


def mutation(
    offspring1, offspring2, cur_gen, total_gen, prob_mut=0.3, mut_strength=0.7
):

    o1 = individuals[offspring1]
    o2 = individuals[offspring2]

    indices = [offspring1, offspring2]
    mask = np.ones(len(individuals), dtype=bool)
    mask[indices] = False

    alternatives = individuals[mask, ...]

    it = np.nditer([o1, o2], flags=["multi_index"], op_flags=["readwrite"])
    for wx1, wx2 in it:
        if wx1 != 0 and random.random() < prob_mut:

            select = random.sample(list(alternatives), 2)

            o1[it.multi_index[0], it.multi_index[1]] = wx1 + mut_strength * (
                1 - cur_gen / total_gen
            ) * (
                select[0][it.multi_index[0], it.multi_index[1]]
                - select[1][it.multi_index[0], it.multi_index[1]]
            )

        if wx2 != 0 and random.random() < prob_mut:

            select = random.sample(list(alternatives), 2)

            o2[it.multi_index[0], it.multi_index[1]] = wx1 + mut_strength * (
                1 - cur_gen / total_gen
            ) * (
                select[0][it.multi_index[0], it.multi_index[1]]
                - select[1][it.multi_index[0], it.multi_index[1]]
            )

    return offspring1, offspring2

    # print (it.multi_index)
    # print(wx1, wx2)
    # print(wx1[0], wx2[0])

    # for j in range (np.shape(offspring1)[1]):
    #   for k in range (np.shape(offspring1)[0]):
    #       if offspring1(k, j) != 0 and random.random < prob_mut:
    #               for ind in individuals:


# print(individuals[w1], "\n", individuals[w2], "\n")
offspring1, offspring2 = crossover(w1, w2)
# print(individuals[w1],"\n", individuals[w2], "\n")
mut1, mut2 = mutation(offspring1, offspring2, 1, 10)
