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

    def _next_gen_():
        """Run one generation of an EA. Change nothing about the parameters."""
        pass

    def _next_iteration_():
        """Run one iteration (a number of generations) of EA."""
        pass

    def _run_interruption_():
        """Run interruptions in between iterations.
        You can update parameters/EA specific classes here."""
        pass


class baseDecompositionEA(baseEA):
    """This class provides the basic structure for decomposition based Evolutionary
    algorithms, such as RVEA or NSGA-III"""

    def __init__(self, population: "Population", problem: "baseProblem"):
        """
        Initialize RVEA.

        This will set up the parameters of RVEA, create Reference Vectors, and
        (as of Feb 2019) run the first iteration of RVEA.

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
        self.params = self.set_params()
        self.reference_vectors = self.create_reference_vectors(
            self.params.lattice_resolution
        )
        self.select = None
        print("Using baseDecompositionEA init")

    def _next_gen_(self, population: "Population", problem: "baseProblem"):
        """Run one generation of decomposition based EA.

        This method leaves method.params unchanged.
        Intended to be used by next_iteration.

        Parameters
        ----------
        population : Population
        problem : baseProblem

        """
        offspring = population.mate()
        population.add(offspring, problem)
        # APD Based selection
        penalty_factor = (
            (self.params.gen_count / self.params.generations) ** self.params.Alpha
        ) * problem.num_of_objectives
        select = self.select(population.fitness, self.reference_vectors, penalty_factor)
        population.keep(select)

    def select(self):
        """Describe a selection mechanism. Return indices of selected individuals."""
