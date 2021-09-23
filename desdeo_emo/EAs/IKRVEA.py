from typing import Dict, Union

from desdeo_emo.EAs import RVEA
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.model_management import ikrvea_mm
# from desdeo_emo.selection.APD_Select import APD_Select
from desdeo_emo.selection.APD_Select_constraints import APD_Select
from desdeo_emo.selection.oAPD import Optimistic_APD_Select
from desdeo_emo.selection.robust_APD import robust_APD_Select
from desdeo_problem import MOProblem


class IK_RVEA(RVEA):
    """The python version Interactive Kriging-assisted reference vector guieded evolutionary algorithm (IK-RVEA).

    Most of the relevant code is contained in the super class. This class just assigns
    the APD selection operator, and the model management to BaseDecompositionEA.

    NOTE: The APD (from RVEA) function had to be slightly modified to accomodate for the fact that
    this version of the algorithm is interactive, and does not have a set termination
    criteria. There is a time component in the APD penalty function formula of the type:
    (t/t_max)^alpha. As there is no set t_max, the formula has been changed. See below,
    the documentation for the argument: penalty_time_component

    See the details of IKRVEA in the following paper
    'P. Aghaei Pour, T. Rodemann, J. Hakanen, and K. Miettinen, “Surrogate assisted interactive
    multiobjective optimization in energy system design of buildings,” 
    Optimization and Engineering, 2021.'
    
    See the details of RVEA in the following paper
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

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
    alpha : float, optional
        The alpha parameter in the APD selection mechanism. Read paper for details.
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
    number_of_update: int, optional
        The number of solutions that are selected for true function evaluations, by default 10.
        This is not a hard limit and is set based on amount of time the user has and how long each true evaluation takes.
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    penalty_time_component: Union[str, float], optional
        The APD formula had to be slightly changed.
        If penalty_time_component is a float between [0, 1], (t/t_max) is replaced by
        that constant for the entire algorithm.
        If penalty_time_component is "original", the original intent of the paper is
        followed and (t/t_max) is calculated as
        (current generation count/total number of generations).
        If penalty_time_component is "function_count", (t/t_max) is calculated as
        (current function evaluation count/total number of function evaluations)
        If penalty_time_component is "interactive", (t/t_max)  is calculated as
        (Current gen count within an iteration/Total gen count within an iteration).
        Hence, time penalty is always zero at the beginning of each iteration, and one
        at the end of each iteration.
        Note: If the penalty_time_component ever exceeds one, the value one is used as
        the penalty_time_component.
        If no value is provided, an appropriate default is selected.
        If `interact` is true, penalty_time_component is "interactive" by default.
        If `interact` is false, but `total_function_evaluations` is provided,
        penalty_time_component is "function_count" by default.
        If `interact` is false, but `total_function_evaluations` is not provided,
        penalty_time_component is "original" by default.
    """

    def __init__(
            self,
            problem: MOProblem,
            population_size: int = None,
            population_params: Dict = None,
            initial_population: Population = None,
            alpha: float = 2,
            lattice_resolution: int = None,
            selection_type: str = None,
            a_priori: bool = False,
            interact: bool = True,
            use_surrogates: bool = False,
            n_iterations: int = 10,
            n_gen_per_iter: int = 100,
            number_of_update: int = 10,
            total_function_evaluations: int = 0,
            time_penalty_component: Union[str, float] = None,
            
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
        )

        self.number_of_update = number_of_update # number of solutions that we use to update surrogates


    def iterate(self, ref):

        super().iterate(ref)
        updated_problem = ikrvea_mm(
        ref.response.values,
        self.population.individuals,
        self.population.objectives,
        self.population.uncertainity,
        self.population.problem,
        self.number_of_update)
        self.population.problem = updated_problem
