from pyrvea.EAs.RVEA import RVEA
from pyrvea.OtherTools.ReferenceVectors import ReferenceVectors
from typing import TYPE_CHECKING
from pyrvea.Problem.testProblem import testProblem
from pyrvea.Population.create_individuals import create_new_individuals
import numpy as np

if TYPE_CHECKING:
    from pyrvea.Population.Population import Population


class slowRVEA(RVEA):
    """RVEA variant that impliments slow reference vector movement."""

    def __init__(self, population: "Population", **ea_parameters):
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
        # if population.individuals.shape[0] == 0:
        #     create_new_individuals(pop_size=self.params["population_size"])
        # # print("Using BaseDecompositionEA init")
        # self._next_iteration(population)

    def set_params(
        self,
        population: "Population",
        generations_per_iteration: int = 100,
        iterations: int = 10,
        Alpha: float = 2,
        plotting: bool = True,
        logging: bool = False,
        logfile = None,
        ref_point: list = None,
        old_point: list = None,
    ):
        """Set up the parameters. Save in RVEA.params. Note, this should be
        changed to align with the current structure.

        Parameters
        ----------
        population : Population
            Population object
        Alpha : float
            The alpha parameter of APD selection.
        plotting : bool
            Useless really.
        Returns
        -------

        """
        ref_vectors = ReferenceVectors(
            number_of_objectives=population.problem.num_of_objectives,
            creation_type="Sparse_Focused",
            ref_point=old_point,
        )
        if ref_point is None:
            ref_point = ref_vectors.values[0]
        rveaparams = {
            "reference_vectors": ref_vectors,
            "population_size": ref_vectors.number_of_vectors,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "Alpha": Alpha,
            "ploton": plotting,
            "logging": logging,
            "logfile": logfile,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
            "current_total_gen_count": 0,
            "total_generations": iterations * generations_per_iteration,
            "ref_point": ref_point,
        }
        return rveaparams

    def _run_interruption(self, population: "Population"):
        self.params["reference_vectors"].slow_interactive_adapt(
            self.params["ref_point"]
        )