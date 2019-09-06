from typing import Dict

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.NSGAIII_select import NSGAIII_select
from desdeo_problem.Problem import MOProblem


class NSGAIII(BaseDecompositionEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    [description]
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        n_survive: int = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
        )
        selection_operator = NSGAIII_select(self.population, n_survive)
        self.selection_operator = selection_operator
        self._next_iteration()
