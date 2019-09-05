from pyrvea.Population.Population import Population
from pyrvea.Selection.tournament_select import tour_select
import numpy as np


class TournamentEA:
    def __init__(self, population: "Population", ea_parameters):
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

        self.params = self.set_params(population, **ea_parameters)

    def set_params(
        self,
        population: "Population",
        tournament_size: int = 5,
        target_pop_size: int = 500,
        generations_per_iteration: int = 10,
        iterations: int = 10,
        prob_crossover: float = 0.9,
        prob_mutation: float = 0.3,
        min_fitness: float = 0.001,
    ):
        """Set up the parameters.

        Parameters
        ----------
        population : Population
            Population object.
        tournament_size : int
            Number of participants in tournament selection.
        target_pop_size : int
            Desired population size.
        generations_per_iteration : int
            Number of generations per iteration.
        iterations : int
            Total number of iterations.
        prob_crossover : float
            Probability of crossover occurring.
        prob_mutation : float
            Probability of mutation occurring.
        min_fitness : float
            If error of the best solution < min_fitness, stop evolution.

        Returns
        -------
        params : dict
            Parameters for the algorithm.

        """

        params = {
            "population": population,
            "tournament_size": tournament_size,
            "target_pop_size": target_pop_size,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "total_generations": iterations * generations_per_iteration,
            "current_iteration_gen_count": 0,
            "current_total_gen_count": 0,
            "current_iteration_count": 0,
            "prob_crossover": prob_crossover,
            "prob_mutation": prob_mutation,
            "min_fitness": min_fitness,
        }
        return params

    def _next_iteration(self, population: "Population"):
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.

        Parameters
        ----------
        population : "Population"
            Contains current population
        """
        self.params["current_iteration_gen_count"] = 1
        while self.continue_iteration():
            self._next_gen(population)
            self.params["current_iteration_gen_count"] += 1
            self.params["current_total_gen_count"] += 1
        self.params["current_iteration_count"] += 1

    def _next_gen(self, population: "Population"):
        """Run one generation of decomposition based EA.

        This method leaves method.params unchanged. Intended to be used by
        next_iteration.

        Parameters
        ----------
        population: "Population"
            Population object
        """

        selected = self.select(population)
        offspring = population.mate(mating_pop=selected, params=self.params)
        population.delete(np.arange(len(population.individuals)))
        population.add(offspring)

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

    def _run_interruption(self, population: "Population"):
        """Run the interruption phase of PPGA.

        Use this phase to make changes to PPGA.params or other objects.

        Parameters
        ----------
        population : Population
        """

        pass

    def select(self, population) -> list:
        """Select parents for recombination using tournament selection.
        Chooses two parents, which are needed for crossover.

        Parameters
        ----------
        population : Population
            Contains the current population and problem
            information.

        Returns
        -------
        parents : list
            List of indices of individuals to be selected.
        """
        parents = []
        for i in range(int(self.params["target_pop_size"] / 2)):
            parents.append(
                [
                    tour_select(
                        population.fitness, self.params["tournament_size"]
                    ),
                    tour_select(
                        population.fitness, self.params["tournament_size"]
                    ),
                ]
            )
        return parents
