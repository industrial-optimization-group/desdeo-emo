import numpy as np
from random import choice, sample


def mutate(
    offspring,
    individuals,
    params,
    *args
):
    """Perform BioGP mutation functions.

    Standard mutation:
    Randomly select and regrow a subtree of an individual.

    Small mutation:
    Randomly select a node within a tree and replace it with either a function of the
    same arity,
    or another value from the terminal set.

    Mono parental:
    Randomly swap two subtrees within the same individual.

    Parameters
    ----------
    offspring : list
        List of individuals to mutate.
    individuals : list
        List of all individuals.
    params : dict
        Parameters for breeding. If None, use defaults.

    """

    prob_mut = params.get("prob_mutation", 0.3)
    prob_stand = 1/3 * prob_mut
    prob_point = 1/3 * prob_mut
    prob_mono = prob_mut - prob_stand - prob_point
    prob_replace = prob_mut
    r = np.random.rand()

    for ind in offspring:
        if r <= prob_stand:
            # Standard mutation
            #
            # This picks a random subtree anywhere within the tree
            rand_node = choice(ind.nodes[1:])
            tree = ind.grow_tree(method="grow", depth=rand_node.depth, ind=rand_node)
            rand_node.value = tree.value
            rand_node.roots = tree.roots

            # This picks a whole subtree at depth=1 under the linear node
            # rand_subtree = np.random.randint(len(ind.roots))
            # del ind.roots[rand_subtree]
            # ind.grow_tree(method="grow", ind=ind)

            ind.nodes = ind.get_sub_nodes()

        elif r <= prob_point + prob_stand:
            # Small mutation
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
            # Mono parental
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
