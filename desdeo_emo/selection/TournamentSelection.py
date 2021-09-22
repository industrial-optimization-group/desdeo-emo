import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class TournamentSelection(SelectionBase):
    """
        Tournament selection operator.

        Parameters
        ----------
        pop : Population
            The population of individuals
        tournament_size : int
            Size of the tournament.
    """
    def __init__(self, pop, tournament_size):
        # initialize
        self.fitness = pop.fitness
        self.pop_size = pop.pop_size
        self.tournament_size = tournament_size


    def do(self, pop) -> List[int]:
        """ Performs tournament selections and returns the parents.
        Parameters
        ----------
        pop : Population
            The current population.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        self.fitness = pop.fitness
        self.pop_size= pop.pop_size
        parents = []
        for i in range(int(self.pop_size)): 
            parents.append(
                np.asarray(
                    self._tour_select(), 
                    self._tour_select(),
            ))
        return parents


    def _tour_select(self):
        """Tournament selection. Choose number of individuals to participate
        and select the one with the best fitness.

        Parameters
        ----------
        fitness : array_like
            An array of each individual's fitness.
        tournament_size : int
            Number of participants in the tournament.

        Returns
        -------
        int
            The index of the best individual.
        """
        fitness = self.fitness[:,0]
        aspirants = np.random.choice(len(fitness)-1, self.tournament_size, replace=False)
        chosen = []
        for ind in aspirants:
            chosen.append([ind, fitness[ind]])
        chosen.sort(key=lambda x: x[1])

        return chosen[0][0]
