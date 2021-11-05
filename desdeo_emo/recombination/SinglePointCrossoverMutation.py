import numpy as np
from random import shuffle

class SinglePoint_Xover:
    """Simple single point crossover and mutation.

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

    def crossover(self, ind_0, ind_1):
        """
        Single point crossover.
        Args:
            ind_0: individual_0
            ind_1: individual_1
        Ret:
            new_0, new_1: the individuals generatd after crossover.
        """
        assert(len(ind_0) == len(ind_1))

        point = np.random.randint(len(ind_0))
#         new_0, new_1 = np.zeros(len(ind_0)),  np.zeros(len(ind_0))
        new_0 = np.hstack((ind_0[:point], ind_1[point:]))
        new_1 = np.hstack((ind_1[:point], ind_0[point:]))

        assert(len(new_0) == len(ind_0))
        return new_0, new_1

    def mutation(self, indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi

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

            
            offspring[i], offspring[i + 1] = self.crossover(mating_pop[i],mating_pop[i+1])
            offspring = offspring.round()
        return offspring

class SinglePoint_Mutation:
    """Simple single point crossover and mutation.

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

    def mutation(self, indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi

    def do(self, offspring: np.ndarray):
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
        offspring_size, num_var = offspring.shape
        k = np.random.randint(0,num_var, offspring_size)
       
        for i in range(offspring_size):
            offspring[i,k[i]] = 1 - offspring[i,k[i]]
        
        return offspring
