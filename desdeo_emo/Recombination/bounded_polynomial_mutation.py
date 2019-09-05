import numpy as np


def mutate(offspring, individuals, params, lower_limits, upper_limits):
    """Bounded polynomial mutation.

    Parameters
    ----------
    offspring : List
        List of offspring to mutate.
    individuals : List
        List of all individuals.
    params : dict
        Parameters used for breeding.
    lower_limits : float
        Problem lower bounds.
    upper_limits : float
        Problem upper bounds.

    """
    dis_mutation = params.get("dis_mutation", 20)

    prob_mutation = 1 / np.array(individuals).shape[1]

    min_val = np.ones_like(offspring) * lower_limits
    max_val = np.ones_like(offspring) * upper_limits
    k = np.random.random(offspring.shape)
    miu = np.random.random(offspring.shape)
    temp = np.logical_and((k <= prob_mutation), (miu < 0.5))
    offspring_scaled = (offspring - min_val) / (max_val - min_val)
    offspring[temp] = offspring[temp] + (
        (max_val[temp] - min_val[temp])
        * (
            (
                2 * miu[temp]
                + (1 - 2 * miu[temp])
                * (1 - offspring_scaled[temp]) ** (dis_mutation + 1)
            )
            ** (1 / (dis_mutation + 1))
            - 1
        )
    )
    temp = np.logical_and((k <= prob_mutation), (miu >= 0.5))
    offspring[temp] = offspring[temp] + (
        (max_val[temp] - min_val[temp])
        * (
            1
            - (
                2 * (1 - miu[temp])
                + 2 * (miu[temp] - 0.5) * offspring_scaled[temp] ** (dis_mutation + 1)
            )
            ** (1 / (dis_mutation + 1))
        )
    )
    offspring[offspring > max_val] = max_val[offspring > max_val]
    offspring[offspring < min_val] = min_val[offspring < min_val]
