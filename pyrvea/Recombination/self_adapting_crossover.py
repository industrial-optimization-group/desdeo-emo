import numpy as np


def crossover(population, w1, w2, prob_xover=0.8):

    individuals = population.individuals

    for i in range(np.shape(individuals[w1])[1]):
        if np.random.random() < prob_xover:
            tmp = np.copy(individuals[w1][:, i])
            individuals[w1][:, i] = individuals[w2][:, i]
            individuals[w2][:, i] = tmp

    return w1, w2
