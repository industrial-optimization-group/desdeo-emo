import numpy as np
from random import shuffle
#from numba import njit


class SBX_xover:
    """Simulated binary crossover.

    Parameters
        ----------
        ProC : float, optional
            [description], by default 1
        DisC : float, optional
            [description], by default 30
    """

    def __init__(self, ProC: float = 1, DisC: float = 30):
        """[summary]


        """
        self.ProC = ProC
        self.DisC = DisC

    def do(self, pop: np.ndarray, mating_pop_ids: list = None) -> np.ndarray:
        """Consecutive members of mating_pop_ids are crossed over
            in pairs. Example: if mating_pop_ids = [0, 2, 3, 6, 5] then the individuals
            are crossover as: [0, 2], [3, 6], [5, 0]. Note: if the number of elements
            is odd, the last individual is crossed over with the first one.

        Parameters
        ----------
        pop : np.ndarray
            Array of all individuals
        mating_pop_ids : list, optional
            Indices of population members to mate, by default None, which shuffles and
                mates whole population

        Returns
        -------
        np.ndarray
            The offspring produced as a result of crossover.
        """
        pop_size, num_var = pop.shape
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        mating_pop = pop[shuffled_ids]
        mate_size = len(shuffled_ids)
        if len(shuffled_ids) % 2 == 1:
            # Maybe it should be pop_size-1?
            mating_pop = np.vstack((mating_pop, mating_pop[0]))
            mate_size = mate_size + 1
        # The rest closely follows the matlab code.
        offspring = np.zeros_like(mating_pop)  # empty_like() more efficient?
        for i in range(0, mate_size, 2):
            beta = np.zeros(num_var)
            miu = np.random.rand(num_var)
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (self.DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (self.DisC + 1))
            beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
            beta[np.random.rand(num_var) > self.ProC] = 1  # It was in matlab code
            avg = (mating_pop[i] + mating_pop[i + 1]) / 2
            diff = (mating_pop[i] - mating_pop[i + 1]) / 2
            offspring[i] = avg + beta * diff
            offspring[i + 1] = avg - beta * diff
        return offspring


# TODO: Make sure the following works correctly, then replace the code above.
"""@njit
def create_offsprings(pop:np.ndarray, mate_size:int,ProC: float=1, DisC: float=30):
    pop_size, num_var = pop.shape
    beta = np.zeros(num_var)
    offspring = np.zeros_like(pop)
    for i in range(0, mate_size, 2):
        beta[:] = 0
        miu = np.random.rand(num_var)

        beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
        beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))

        beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
        beta[np.random.rand(num_var) > ProC] = 1  # It was in matlab code

        avg = (pop[i] + pop[i + 1]) / 2
        diff = (pop[i] - pop[i + 1]) / 2
        offspring[i] = avg + beta * diff
        offspring[i + 1] = avg - beta * diff
    return offspring"""
