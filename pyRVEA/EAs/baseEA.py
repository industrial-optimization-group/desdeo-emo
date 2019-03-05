from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population
    from pyRVEA.Problem.baseProblem import baseProblem


class baseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__():
        """Initialize EA here. Set up parameters, create EA specific objects."""
        pass

    def set_params(self):
        """Set up the parameters. Save in self.params"""

        pass

    def _next_gen():
        """Run one generation of an EA. Change nothing about the parameters."""
        pass

    def _next_iteration():
        """Run one iteration (a number of generations) of EA."""
        pass

    def _run_interruption():
        """Run interruptions in between iterations.
        You can update parameters/EA specific classes here."""
        pass


class baseDecompositionEA(baseEA):
    """This class provides the basic structure for decomposition based Evolutionary
    algorithms, such as RVEA or NSGA-III"""

    def __init__(self, population: "Population", EA_parameters: dict = None):
        """
        Initialize a Base Decomposition EA.

        This will call methods to set up the parameters of RVEA,
        create Reference Vectors, and (as of Feb 2019) run the first iteration of RVEA.

        Parameters
        ----------
        population : Population
            This variable is updated as evolution takes place
        problem : baseProblem
            Contains the details of the problem.

        Returns
        -------
        Population
            Returns the Population after evolution.

        """
        self.params = self.set_params(population, EA_parameters)
        # print("Using baseDecompositionEA init")
        self._next_iteration(population)

    def _next_gen(self, population: "Population"):
        """Run one generation of decomposition based EA.

        This method leaves method.params unchanged.
        Intended to be used by next_iteration.


        Parameters
        ----------
        population : Population

        """
        offspring = population.mate()
        population.add(offspring)
        selected = self.select(population)
        population.keep(selected)

    def select(self, population):
        """Describe a selection mechanism. Return indices of selected individuals.

        Parameters
        ----------
        population : Population
            Contains the current population and problem information.

        Returns
        -------
        list or numpy.ndarray
            Return indices of selected individuals.
        """
        pass

    def continueiteration(self):
        """Checks whether the current iteration should be continued or not.

        Returns
        -------
        bool
            True if iteration to be continued. False otherwise.
        """
        return self.params["current_iteration_gen_count"] < self.params["generations"]
