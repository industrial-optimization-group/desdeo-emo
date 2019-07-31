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

        elif r <= prob_point + prob_stand:
            # Point mutation

            for node in ind.nodes:
                if np.random.rand() < prob_replace and node.__class__.__name__ == "TerminalNode":
                    node.value = choice(node.terminal_set)
                elif np.random.rand() < prob_replace and node.__class__.__name__ == "FunctionNode":
                    value = choice(node.function_set)
                    while node.value.__code__.co_argcount != value.__code__.co_argcount:
                        value = choice(node.function_set)
                    node.value = value

        elif r <= prob_mono + prob_point + prob_stand:
            # Mono parental xover
            swap_nodes = sample(ind.nodes, 2)
            tmp = swap_nodes[0]
            swap_nodes[0] = swap_nodes[1]
            swap_nodes[1] = tmp

        else:
            pass
