import numpy as np


def ppga_crossover(w1, w2, prob_crossover=0.8):
    """Randomly exchange nodes between two individuals.

    Parameters
    ----------
    w1 : np.ndarray
        The first individual
    w2 : np.ndarray
        The second individual
    prob_crossover : float
        The probability for the crossover
    """

    for i in range(np.shape(w1)[1]):
        if np.random.random() < prob_crossover:
            tmp = np.copy(w1[:, i])
            w1[:, i] = w2[:, i]
            w2[:, i] = tmp

    return w1, w2
