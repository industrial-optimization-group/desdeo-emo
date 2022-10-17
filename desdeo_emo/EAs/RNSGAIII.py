from typing import Dict

import numpy as np

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.RNSGAIII_select import RNSGAIII_select
from desdeo_problem import MOProblem

from scipy.special import comb

class RNSGAIII(BaseDecompositionEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    Most of the relevant code is contained in the super class. This class just assigns
    the NSGAIII selection operator to BaseDecompositionEA.

    Parameters
    ----------
    problem : MOProblem
        The problem class object specifying the details of the problem.
    population_size : int, optional
        The desired population size, by default None, which sets up a default value
        of population size depending upon the dimensionaly of the problem.
    population_params : Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population : Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    lattice_resolution : int, optional
        The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    selection_type : str, optional
        One of ["mean", "optimistic", "robust"]. To be used in data-driven optimization.
        To be used only with surrogate models which return an "uncertainity" factor.
        Using "mean" is equivalent to using the mean predicted values from the surrogate
        models and is the default case.
        Using "optimistic" results in using (mean - uncertainity) values from the
        the surrogate models as the predicted value (in case of minimization). It is
        (mean + uncertainity for maximization).
        Using "robust" is the opposite of using "optimistic".
    a_priori : bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact : bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    n_iterations : int, optional
        The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter : int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size_per_rp: int,
        ref_points: np.array,
        population_params: Dict = None,
        n_survive: int = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        selection_type: str = None,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        keep_archive: bool = False,
    ):
        self.n_ref_points = ref_points.shape[0]
        self.pop_size_rp = population_size_per_rp
        self.ref_points = ref_points
        temp_lattice_resolution = 0
        temp_number_of_vectors = 0
        while True:
            temp_lattice_resolution += 1
            temp_number_of_vectors = comb(
                temp_lattice_resolution + problem.n_of_objectives - 1,
                problem.n_of_objectives - 1,
                exact=True,
            )
            if temp_number_of_vectors > population_size_per_rp:
                break
        lattice_resolution = temp_lattice_resolution - 1
        pop_size = (temp_number_of_vectors * self.n_ref_points) + problem.n_of_objectives
        
        super().__init__(
            problem=problem,
            population_size=pop_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            keep_archive=keep_archive,
        )
        self.selection_type = selection_type
        selection_operator = RNSGAIII_select(
            self.population, self.pop_size_rp, self.ref_points, n_survive, selection_type=selection_type,        )
        self.selection_operator = selection_operator
