import numpy as np


def mutate(
    offspring,
    individuals,
    params,
    *args
):
    """ Perform self-adapting mutation over offspring.

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
        cur_gen = params["current_total_gen_count"]
        total_gen = params["total_generations"]
        prob_mutation = params["prob_mutation"]
        mut_strength = params["mut_strength"]

    except KeyError:
        cur_gen = 1
        total_gen = 10
        prob_mutation = 0.3
        mut_strength = 0.7

    alternatives = np.copy(np.array(individuals)[:, 1:, :])

    for ind in offspring:

        connections = ind[1:, :].size

        # Method: Self adapting
        # Choose two random individuals and a random number of connections,
        # mutate offspring based on current gen and connections of two randomly chosen individuals

        # Randomly select two individuals with current match active (=non-zero)
        select = alternatives[
            np.random.choice(
                np.nonzero(alternatives)[
                    0
                ],
                2,
            )
        ]

        mut = np.random.choice(connections, np.random.binomial(connections, prob_mutation), replace=False)
        ind[1:, :].ravel()[mut] = ind[1:, :].ravel()[mut] + mut_strength * (
                    1 - cur_gen / total_gen
                ) * (select[1].ravel()[mut] - select[0].ravel()[mut])
