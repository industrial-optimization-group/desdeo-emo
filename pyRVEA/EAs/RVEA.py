"""The python version reference vector guided evolutionary algorithm.

See the details of RVEA in the following paper

R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
A Reference Vector Guided Evolutionary Algorithm for Many-objective
Optimization, IEEE Transactions on Evolutionary Computation, 2016

The source code of cRVEA is implemented by Bhupinder Saini

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi
Project researcher at University of Jyväskylä.
"""


from typing import TYPE_CHECKING

from pyRVEA.Selection.APD_select import APD_select
from pyRVEA.EAs.baseEA import baseDecompositionEA
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population
    from pyRVEA.Problem.baseProblem import baseProblem
    from pyRVEA.allclasses import Parameters


def rvea(
    population: "Population",
    problem: "baseProblem",
    parameters: "Parameters",
    reference_vectors: "ReferenceVectors",
):
    """
    Run RVEA.

    This only conducts reproduction and selection. Reference vector adaptation should
    be done outside. Changes variable population.

    Parameters
    ----------
    population : Population
        This variable is updated as evolution takes place
    problem : Problem
        Contains the details of the problem.
    parameters : Parameters
        Contains the hyper-parameters of RVEA evolution.
    reference_vectors : ReferenceVectors
        Class containing the reference vectors.
    progressbar : tqdm or tqdm_notebook
        An iterable used to display the progress bar.

    Returns
    -------
    Population
        Returns the Population after evolution.

    """
    refV = reference_vectors.neighbouring_angles()
    progress = range(parameters["generations"])
    for gen_count in progress:
        offspring = population.mate()
        population.add(offspring, problem)
        # APD Based selection
        penalty_factor = (
            (gen_count / parameters["generations"]) ** parameters["Alpha"]
        ) * problem.num_of_objectives
        select = APD_select(population.fitness, reference_vectors, penalty_factor, refV)
        population.keep(select)
    return population


class RVEA(baseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    See the details of RVEA in the following paper

    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
    A Reference Vector Guided Evolutionary Algorithm for Many-objective
    Optimization, IEEE Transactions on Evolutionary Computation, 2016

    The source code of pyRVEA is implemented by Bhupinder Saini

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi
    Project researcher at University of Jyväskylä.
    """

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
        self.select = APD_select
        self._next_iteration_()

    def set_params(self):
        """Set up the parameters. Save in RVEA.params"""

        pass

    def create_reference_vectors(self):
        """Create reference vectors."""

        pass

    def _next_iteration_(self, population: "Population", problem: "baseProblem"):
        """Run one iteration of RVEA.

        One iteration consists of a constant or variable number of generations. This
        method leaves RVEA.params unchanged.

        Parameters
        ----------
        population : Population
        problem : baseProblem

        """
        self.params.gen_count = 0
        continuegenerations = True
        while continuegenerations:
            pass

    def _run_interruption_(self, population: "Population", problem: "baseProblem"):
        """Run the interruption phase of RVEA.
        
        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors, conducts interaction with the user.
        
        Parameters
        ----------
        population : Population
        problem : baseProblem
        
        """

        pass
