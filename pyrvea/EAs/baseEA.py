from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrvea.Population.Population import Population


class BaseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__(self):
        """Initialize EA here. Set up parameters, create EA specific objects."""
        pass

    def set_params(self):
        """Set up the parameters. Save in self.params"""
        pass

    def _next_gen(self):
        """Run one generation of an EA. Change nothing about the parameters."""
        pass

    def _next_iteration(self):
        """Run one iteration (a number of generations) of EA."""
        pass

    def _run_interruption(self):
        """Run interruptions in between iterations. You can update parameters/EA
        specific classes here.
        """
        pass


class BaseDecompositionEA(BaseEA):
    """This class provides the basic structure for decomposition based
    Evolutionary algorithms, such as RVEA or NSGA-III.
    """

    def __init__(self, population: "Population", EA_parameters: dict = None):
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
        self.params = self.set_params(population, EA_parameters)
        # print("Using BaseDecompositionEA init")
        self._next_iteration(population)

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
        offspring = population.mate()
        population.add(offspring)
        selected = self.select(population)
        population.delete_or_keep(selected, "keep")

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
        pass

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        pass


class BasePPGA(BaseEA):
    """The base class for evolutionary algorithms using neural networks."""

    def __init__(self, population: "Population", EA_parameters: dict = None):
        """Initialize the base class, call the method to set up the parameters.

        Parameters
        ----------
        population : "Population"
            This variable is updated as evolution takes place
        EA_parameters : dict
            Takes the EA parameters
        """
        self.params = self.set_params(population, EA_parameters)
        # print("Using BaseDecompositionEA init")
        self._next_iteration(population)

    def initialize_lattice(self):
        pass

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
        self.params["current_iteration_count"] += 1

    def _next_gen(self, population: "Population"):
        """Run one generation of EA.

        This method leaves method.params unchanged. Intended to be used by
        next_iteration.

        Parameters
        ----------
        population: "Population"
            Population object
        """

        ind1, ind2 = self.move_prey()
        offspring = population.mate(ind1, ind2)
        population.add(offspring)
        selected = self.select(population)
        population.delete_or_keep(selected, "keep")

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
        pass

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        pass
