
from typing import Dict
from desdeo_emo.population.Population import Population
from desdeo_problem.Problem import MOProblem
from desdeo_emo.selection.APD_Select import APD_Select
from desdeo_emo.EAs.baseEA import BaseDecompositionEA


class RVEA(BaseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    See the details of RVEA in the following paper

    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

    The source code of pyrvea is implemented by Bhupinder Saini

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi

    Project researcher at University of Jyväskylä.
    """

    def __init__(
        self,
        problem: MOProblem,
        selection_operator: APD_Select,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        selection_parameters: Dict = None,
    ):
        super().__init__(
            problem,
            selection_operator,
            population_size,
            population_params,
            initial_population,
            lattice_resolution,
            a_priori,
            interact,
            n_iterations,
            n_gen_per_iter,
            selection_parameters,
        )
