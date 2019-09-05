import numpy as np
from random import shuffle


def mate(mating_pop, pop, params):
    """Simulated binary crossover.

    Parameters
    ----------
    mating_pop : list
        List of population to mate.
    pop : list
        List of all individuals
    params : dict
        Parameters for breeding. If None, use defaults.

    Returns
    -------
    offspring : List
        The offspring produced as a result of crossover.
    """
    prob_crossover = params.get("prob_crossover", 1)
    dis_crossover = params.get("dis_crossover", 30)

    pop = np.asarray(pop)
    pop_size, num_var = pop.shape

    if mating_pop is None:
        shuffled_ids = list(range(pop_size))
        shuffle(shuffled_ids)
        # Create random pairs from the population for mating
        mating_pop = [
            shuffled_ids[i * 2 : (i + 1) * 2] for i in range(int(len(shuffled_ids) / 2))
        ]

    # The rest closely follows the matlab code.

    offsprings = np.zeros((0, num_var))  # empty_like() more efficient?

    for i in range(len(mating_pop)):
        beta = np.zeros(num_var)
        miu = np.random.rand(num_var)
        beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (dis_crossover + 1))
        beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (dis_crossover + 1))
        beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
        beta[np.random.rand(num_var) > prob_crossover] = 1  # It was in matlab code
        avg = (pop[mating_pop[i][0]] + pop[mating_pop[i][1]]) / 2
        diff = (pop[mating_pop[i][0]] - pop[mating_pop[i][1]]) / 2
        offsprings = np.vstack((offsprings, avg + beta * diff))
        offsprings = np.vstack((offsprings, avg - beta * diff))

    return offsprings
