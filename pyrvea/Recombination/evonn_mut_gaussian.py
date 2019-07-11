import numpy as np
from copy import deepcopy
from random import sample
from math import ceil
from timeit import default_timer as timer


def mutate(
    offspring,
    individuals,
    params,
    *args
):
    """ Perform mutation based on standard deviation on the offspring.

    Parameters
    ----------
    offspring : list
        List of individuals to mutate.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for breeding. If None, use defaults.

    """

    try:
        prob_mutation = params["prob_mutation"]
        std_dev = params["std_dev"]

    except KeyError:

        prob_mutation = 0.3
        std_dev = (5 / 3) * (
            1
            - params["current_total_gen_count"] / params["total_generations"]
        )
        if std_dev < 0:
            std_dev = 0

    for ind in offspring:

        connections = ind[1:, :].size

        # Method : Gaussian
        # Take a random number of connections based on probability and mutate based on
        # standard deviation, calculated once per generation
        # VERY FAST
        #
        mut_val = np.random.normal(0, std_dev, connections)

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        ind[1:, :].ravel()[mut] += ind[1:, :].ravel()[mut] * mut_val[mut]
