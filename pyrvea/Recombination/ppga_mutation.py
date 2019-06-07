import numpy as np
from random import sample


def ppga_mutation(
    alternatives,
    w1,
    w2,
    cur_gen=1,
    total_gen=10,
    prob_mut=0.3,
    mut_strength=0.7,
):
    """Randomly mutate two individuals

    Paremeters
    ----------
    alternatives : np.ndarray
        a list of individuals excluding w1 and w2
    w1 : np.ndarray
        first individual to be mutated
    w2 : np.ndarray
        second individual
    cur_gen : int
        the number of the current generation
    total_gen : int
        max number of generations
    prob_mut : float
        the probability of the mutation happening
    mut_strength : float
        the strength of the mutation constant
    """

    def mutate(offspring, wx):
        """Internal function for the mutation calculation"""

        select = sample(list(alternatives), 2)

        # while select[0][it.multi_index[0], it.multi_index[1]] or select[1][it.multi_index[0], it.multi_index[1]] == 0:
        #     select = sample(list(alternatives), 2)

        # [it.multi_index[0]+1 to ignore bias
        offspring[it.multi_index[0]+1, it.multi_index[1]] = wx + mut_strength * (
            1 - cur_gen / total_gen
        ) * (
            select[0][it.multi_index[0], it.multi_index[1]]
            - select[1][it.multi_index[0], it.multi_index[1]]
        )

        return offspring

    # Iterate over both individuals at the same time to avoid nested looping
    it = np.nditer([w1[1:, :], w2[1:, :]], flags=["multi_index"], op_flags=["readwrite"])

    # wx = individual cells (connections) in the weight matrix
    for wx1, wx2 in it:
        if wx1 != 0 and np.random.random() < prob_mut:

            w1 = mutate(w1, wx1)

        if wx2 != 0 and np.random.random() < prob_mut:

            w2 = mutate(w2, wx2)

    return w1, w2
