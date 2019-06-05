from pygmo import fast_non_dominated_sorting as nds
import numpy as np


def ppga_select(fitness: list, max_rank):

    # Calculating fronts and ranks
    _, _, _, rank = nds(fitness)
    to_be_killed = np.nonzero(rank > max_rank)

    return to_be_killed[0]
