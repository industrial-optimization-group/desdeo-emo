import numpy as np


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

    for ind in offspring:
        rand_subtree = np.random.randint(len(ind.roots))
        del ind.roots[rand_subtree]
        params["population"].problem.grow_tree(max_depth=ind.depth, method="grow", ind=ind)
