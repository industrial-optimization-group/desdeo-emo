from pyrvea.EAs.baseEA import BaseEA
from pyrvea.Population.Population import Population
from pyrvea.Selection.tournament_select import tour_select
from random import sample
from operator import attrgetter
import numpy as np

class bioGP:
    def __init__(self, population: "Population", ea_parameters):
        """Initialize a Base Decomposition EA.

        This will call methods to set up the parameters of RVEA, create
        Reference Vectors, and (as of Feb 2019) run the first iteration of RVEA.

        Parameters
        ----------
        population : "Population"
            This variable is updated as evolution takes place
        EA_parameters : dict
            Takes the EA parameters

        Returns
        -------
        Population:
            Returns the Population after evolution.
        """

        self.params = self.set_params(population, **ea_parameters)
        self._next_iteration(population)

    def set_params(
        self,
        population: "Population",
        tournament_size: int = 5,
        target_pop_size: int = 300,
        generations_per_iteration: int = 10,
        iterations: int = 10,
        prob_crossover: float = 0.5
    ):
        """Set up the parameters. Save in self.params"""
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
            "prob_crossover": prob_crossover
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
        print(min(population.fitness[:, 0]))
        #selected = self.select(population)
        offspring = population.mate(params=self.params)
        population.delete_or_keep(np.arange(len(population.individuals)), "delete")
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
        """Describe a selection mechanism. Return indices of selected
        individuals.

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
        return tour_select(population.individuals, self.params["tournament_size"])

