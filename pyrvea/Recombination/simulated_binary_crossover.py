import numpy as np
from random import shuffle


def crossover(population, prob_xover=1, dis_xover=30):

    pop = population.individuals
    pop_size, num_var = pop.shape[0], pop.shape[1]-1
    shuffled_ids = list(range(pop_size))
    shuffle(shuffled_ids)
    mating_pop = pop[shuffled_ids]

    if pop_size % 2 == 1:
        # Maybe it should be pop_size-1?
        mating_pop = np.vstack((mating_pop, mating_pop[0]))
        pop_size = pop_size + 1

    # The rest closely follows the matlab code.

    offspring = np.zeros_like(mating_pop)  # empty_like() more efficient?

    for i in range(0, pop_size, 2):
        beta = np.zeros(num_var)
        miu = np.random.rand(num_var)
        beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (dis_xover + 1))
        beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (dis_xover + 1))
        beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
        beta[np.random.rand(num_var) > prob_xover] = 1  # It was in matlab code
        avg = (mating_pop[i] + mating_pop[i + 1]) / 2
        diff = (mating_pop[i] - mating_pop[i + 1]) / 2
        offspring[i] = avg + beta * diff
        offspring[i + 1] = avg - beta * diff

    return offspring
