import numpy as np
from random import choice, sample


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
    prob_mut = params["prob_mutation"]
    prob_stand = 1/3 * prob_mut
    prob_point = 1/3 * prob_mut
    prob_mono = prob_mut - prob_stand - prob_point
    prob_replace = prob_mut
    r = np.random.rand()

    for ind in offspring:
        if r <= prob_stand:
            # Standard mutation
            rand_subtree = np.random.randint(len(ind.roots))
            del ind.roots[rand_subtree]
            ind.grow_tree(method="grow", ind=ind)
            ind.nodes = ind.get_sub_nodes()
        elif r <= prob_point + prob_stand:
            # Point mutation
            for node in ind.nodes[1:]:
                if np.random.rand() < prob_replace and callable(node.value):
                    value = choice(node.function_set)
                    while node.value.__code__.co_argcount != value.__code__.co_argcount:
                        value = choice(node.function_set)
                    node.value = value
                elif np.random.rand() < prob_replace:
                    node.value = choice(node.terminal_set)
            ind.nodes = ind.get_sub_nodes()

        elif r <= prob_mono + prob_point + prob_stand:
            # Mono parental xover
            swap_nodes = sample(ind.nodes[1:], 2)
            tmp_value = swap_nodes[0].value
            tmp_roots = swap_nodes[0].roots
            swap_nodes[0].value = swap_nodes[1].value
            swap_nodes[0].roots = swap_nodes[1].roots
            swap_nodes[1].value = tmp_value
            swap_nodes[1].roots = tmp_roots
            ind.nodes = ind.get_sub_nodes()

        else:
            pass
