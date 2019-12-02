from desdeo_emo.population.Population import Population
from desdeo_emo.EAs.BaseEA import BaseEA, eaError
from desdeo_emo.selection.tournament_select import tour_select
import numpy as np


class TournamentEA(BaseEA):
    def __init__(
        self,
        problem,
        initial_population: Population,
        n_gen_per_iter: int = 10,
        n_iterations: int = 10,
        tournament_size: int = 5,
        population_size: int = 500,
    ):
        """Run generations of evolutionary algorithm using tournament selection.

        Parameters
        ----------
        population : "Population"
            This variable is updated as evolution takes place
        ea_parameters : dict
            Takes the EA parameters.

        Returns
        -------
        Population:
            Returns the Population after evolution.
        """
        super().__init__(n_gen_per_iter=n_gen_per_iter, n_iterations=n_iterations)
        if initial_population is None:
            msg = "Provide initial population"
            raise eaError(msg)
        self.population = initial_population
        self.target_pop_size = population_size
        self.tournament_size = tournament_size

    def _next_gen(self):
        """Run one generation of decomposition based EA.

        This method leaves method.params unchanged. Intended to be used by
        next_iteration.

        Parameters
        ----------
        population: "Population"
            Population object
        """

        selected = self.select()
        offspring = self.population.mate(mating_individuals=selected)
        self.population.delete(np.arange(len(self.population.individuals)))
        self.population.add(offspring)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]

    def select(self) -> list:
        """Select parents for recombination using tournament selection.
        Chooses two parents, which are needed for crossover.

        Returns
        -------
        parents : list
            List of indices of individuals to be selected.
        """
        parents = []
        for i in range(int(self.target_pop_size / 2)):
            parents.append(
                [
                    tour_select(self.population.fitness[:, 0], self.tournament_size),
                    tour_select(self.population.fitness[:, 0], self.tournament_size),
                ]
            )
        return parents
