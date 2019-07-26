from pyrvea.EAs.baseEA import BaseEA
from pyrvea.Population.Population import Population
from random import sample


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
        # print("Using BaseDecompositionEA init")
        self._next_iteration(population)

    def set_params(self, population: "Population", tournament_size: int = 5):
        """Set up the parameters. Save in self.params"""
        params = {"population": population, "tournament_size": tournament_size}
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
        population.delete_or_keep(selected, "keep")
        offspring = population.mate(mating_pop=selected, params=self.params)
        population.add(offspring)

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

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
        list
            List of indices of individuals to be selected.
        """
        chosen = []
        for i in range(self.params["tournament_size"]):

            aspirants = sample(population.individuals, int(len(population.individuals)*0.1))
            chosen.append(min(aspirants, key=aspirants.fitness[0]))

        return chosen