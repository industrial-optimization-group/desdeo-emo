import numpy as np


def tour_select(fitness, tournament_size):
    """Tournament selection. Choose number of individuals to participate
    and select the one with the best fitness.

    Parameters
    ----------
    fitness : array_like
        An array of each individual's fitness.
    tournament_size : int
        Number of participants in the tournament.

    Returns
    -------
    int
        The index of the best individual.
    """
    aspirants = np.random.choice(len(fitness)-1, tournament_size, replace=False)
    chosen = []
    for ind in aspirants:
        chosen.append([ind, fitness[ind]])
    chosen.sort(key=lambda x: x[1])

    return chosen[0][0]
