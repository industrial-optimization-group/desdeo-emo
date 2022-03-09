from typing import Dict

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.NSGAIII_select import NSGAIII_select
from desdeo_problem import MOProblem


class NSGAIII(BaseDecompositionEA):
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
        population_size: int = None,
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
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
        )
        self.selection_type = selection_type
        selection_operator = NSGAIII_select(
            self.population, n_survive, selection_type=selection_type
        )
        self.selection_operator = selection_operator
