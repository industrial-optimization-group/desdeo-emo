from typing import Dict, Union

from desdeo_problem import ScalarConstraint
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Variable import variable_builder
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs import NSGAIII
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, eaError
from desdeo_emo.population.Population import Population

# from desdeo_emo.selection.APD_Select import APD_Select
from desdeo_emo.selection.APD_Select_constraints import APD_Select
from desdeo_emo.selection.oAPD import Optimistic_APD_Select
from desdeo_emo.selection.robust_APD import robust_APD_Select
from desdeo_problem.Problem import MOProblem

import numpy as np
import pandas as pd

class RVEA(BaseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    Most of the relevant code is contained in the super class. This class just assigns
    the APD selection operator to BaseDecompositionEA.

    NOTE: The APD function had to be slightly modified to accomodate for the fact that
    this version of the algorithm is interactive, and does not have a set termination
    criteria. There is a time component in the APD penalty function formula of the type:
    (t/t_max)^alpha. As there is no set t_max, the formula has been changed. See below,
    the documentation for the argument: penalty_time_component

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
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
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
        self.time_penalty_component = time_penalty_component
        time_penalty_component_options = ["original", "function_count", "interactive"]
        if time_penalty_component is None:
            if interact is True:
                time_penalty_component = "interactive"
            elif total_function_evaluations > 0:
                time_penalty_component = "function_count"
            else:
                time_penalty_component = "original"
        if not (type(time_penalty_component) is float or str):
            msg = (
                f"type(time_penalty_component) should be float or str"
                f"Provided type: {type(time_penalty_component)}"
            )
            eaError(msg)
        if type(time_penalty_component) is float:
            if (time_penalty_component <= 0) or (time_penalty_component >= 1):
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
            time_penalty_function = self._time_penalty_constant
        if type(time_penalty_component) is str:
            if time_penalty_component == "original":
                time_penalty_function = self._time_penalty_original
            elif time_penalty_component == "function_count":
                time_penalty_function = self._time_penalty_function_count
            elif time_penalty_component == "interactive":
                time_penalty_function = self._time_penalty_interactive
            else:
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.selection_type = selection_type
        selection_operator = APD_Select(
            pop=self.population,
            time_penalty_function=self.time_penalty_function,
            alpha=alpha,
            selection_type=selection_type,
        )
        self.selection_operator = selection_operator

    def _time_penalty_constant(self):
        """Returns the constant time penalty value.
        """
        return self.time_penalty_component

    def _time_penalty_original(self):
        """Calculates the appropriate time penalty value, by the original formula.
        """
        return self._current_gen_count / self.total_gen_count

    def _time_penalty_interactive(self):
        """Calculates the appropriate time penalty value.
        """
        return self._gen_count_in_curr_iteration / self.n_gen_per_iter

    def _time_penalty_function_count(self):
        """Calculates the appropriate time penalty value.
        """
        return self._function_evaluation_count / self.total_function_evaluations


class oRVEA(RVEA):
    """
    Feature incorporated in the RVEA class using the "selection_type" argument.
    To be depreciated.
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
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
        selection_operator = Optimistic_APD_Select(
            self.population, self.time_penalty_function, alpha
        )
        self.selection_operator = selection_operator


class robust_RVEA(RVEA):
    """
    Feature incorporated in the RVEA class using the "selection_type" argument.
    To be depreciated.
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
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
        selection_operator = robust_APD_Select(
            self.population, self.time_penalty_function, alpha
        )
        self.selection_operator = selection_operator


# testing the method
if __name__ == "__main__":
    # river pollution prob
    """
    # create the problem
    def f_1(x):
        return 4.07 + 2.27 * x[:, 0]


    def f_2(x):
        return 2.60 + 0.03 * x[:, 0] + 0.02 * x[:, 1] + 0.01 / (1.39 - x[:, 0] ** 2) + 0.30 / (1.39 - x[:, 1] ** 2)


    def f_3(x):
        return 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)


    def f_4(x):
        return 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)


    # def f_5(x):
    # return -0.96 + 0.96 / (1.09 - x[:, 1]**2)

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)


    f1 = _ScalarObjective(name="f1", evaluator=f_1, maximize=True)
    f2 = _ScalarObjective(name="f2", evaluator=f_2, maximize=True)
    f3 = _ScalarObjective(name="f3", evaluator=f_3, maximize=True)
    f4 = _ScalarObjective(name="f4", evaluator=f_4, maximize=True)
    f5 = _ScalarObjective(name="f5", evaluator=f_5, maximize=False)

    varsl = variable_builder(["x_1", "x_2"],
                             initial_values=[0.5, 0.5],
                             lower_bounds=[0.3, 0.3],
                             upper_bounds=[1.0, 1.0]
                             )

    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5])

    evolver = RVEA(problem, interact=True, n_iterations=5, n_gen_per_iter=100)

    """

    # test problem from article
    mu = 0.5
    rho = 0.0000078
    Mf = 3
    n = 250
    Jz = 55
    w = np.pi * n / 30

    def Mh(x):
        x = np.atleast_2d(x)
        return (2 / 3) * mu * x[:, 5 - 2] * x[:, 6 - 2] * ((x[:, 3 - 2] ** 3 - x[:, 2 - 2] ** 3) / (x[:, 3 - 2] ** 2 - x[:, 2 - 2] ** 2)) * 0.001

    def th(x):
        return Jz * w / (Mh(x) + Mf)

    # objectives
    def f_1(x):
        x = np.atleast_2d(x)
        return np.pi * (x[:, 3 - 2]**2 - x[:, 2 - 2]**2) * x[:, 4 - 2] * (x[:, 6 - 2] + 1) * rho

    def f_2(th):
        return th

    def f_3(x):
        x = np.atleast_2d(x)
        return x[:, 6 - 2]

    def f_4(x):
        x = np.atleast_2d(x)
        return x[:, 3 - 2]

    def f_5(x):
        x = np.atleast_2d(x)
        return x[:, 5 - 2]

    obj1 = _ScalarObjective("obj1", f_1)
    obj2 = _ScalarObjective("obj2", f_2)
    obj3 = _ScalarObjective("obj3", f_3)
    obj4 = _ScalarObjective("obj4", f_4)
    obj5 = _ScalarObjective("obj5", f_5)

    # variables
    var_names = ["x1", "x2", "x3", "x4", "x5"]  # Make sure that the variable names are meaningful to you.

    initial_values = np.array([70, 100, 2, 800, 5])
    lower_bounds = [60, 90, 1, 600, 1]
    upper_bounds = [80, 110, 3, 1000, 10]
    bounds = np.stack((lower_bounds, upper_bounds))
    variables = variable_builder(var_names, initial_values, upper_bounds=upper_bounds, lower_bounds=lower_bounds)

    # constraints
    Rimin = 60 
    Romax = 110 
    Amin = 1.5 
    deltaR = 20 
    Amax = 3 
    delta = 0.5 
    Lmax = 30 
    Zmax = 10
    Vsrmax = 10000
     # mu = 0.5 
     # rho = 0.0000078 
    s = 1.5 
    Ms = 40 
     # Mf = 3 
     # n = 250 
    Pmax = 1   # 10 
     # Jz = 55 
    tmax = 15 
    Fmax = 1000 

    def S(x):
        x = np.atleast_2d(x)
        return np.pi * (x[:, 3 - 2]**2 - x[:, 2 - 2]**2)

    def Prz(x):
        x = np.atleast_2d(x)
        return x[:, 5 - 2] / S

    def Rsr(x):
        x = np.atleast_2d(x)
        return (2 / 3) * ((x[:, 3 - 2]**3 - x[:, 2 - 2]**3) / (x[:, 3 - 2]**2 - x[:, 2 - 2]**2))

    def Vsr(x):
        x = np.atleast_2d(x)
        return (np.pi * Rsr(x) * n) / 30

     # w = pi * n / 30 
     # Mh = (2 / 3) * mu * x(5 - 1) * x(6 - 1) * ((x(3 - 1) ** 3 - x(2 - 1) ** 3) / (x(3 - 1) ** 2 - x(2 - 1) ** 2)) * 0.001

     # c(1) = -x(2 - 1) + Rimin 
     # c(2) = -Romax + x(3 - 1) 
    def c_1(x):
        x = np.atleast_2d(x)
        return -(x[:, 3 - 2] - x[:, 2 - 2]) + deltaR
     # c(4) = -x(4 - 1) + Amin 
     # c(5) = -Amax + x(4 - 1)

    def c_2(x):
        x = np.atleast_2d(x)
        return -Lmax + (x[:, 6 - 2] + 1) * (x[:, 4 - 2] + delta)

     # c(3) = -Zmax + (x(6 - 1) + 1) 
     # c(8) = -x(6 - 1) + 1

    def c_3(x):
        return -Pmax + Prz(x)

    def c_4(x):
        return -Pmax * Vsrmax + Prz(x) * Vsr(x)

    def c_5(x):
        return -Vsrmax + Vsr(x)

    def c_6(x):
        return -tmax + th(x)

    def c_7(x):
        return -Mh(x) + s * Ms

    def c_8(x):
        return -th(x)

     # c(15) = -x(5 - 1) 
     # c(16) = -Fmax + x(5 - 1)

    cons1 = ScalarConstraint("c_1", 5, 5, c_1)
    cons2 = ScalarConstraint("c_2", 5, 5, c_2)
    cons3 = ScalarConstraint("c_3", 5, 5, c_3)
    cons4 = ScalarConstraint("c_4", 5, 5, c_4)
    cons5 = ScalarConstraint("c_5", 5, 5, c_5)
    cons6 = ScalarConstraint("c_6", 5, 5, c_6)
    cons7 = ScalarConstraint("c_7", 5, 5, c_7)
    cons8 = ScalarConstraint("c_8", 5, 5, c_8)

    # problem
    problem = MOProblem(objectives=[obj1, obj2, obj3, obj4, obj5], variables=variables, constraints=[cons1, cons2, cons3, cons4,
                                                                                             cons5, cons6, cons7, cons8])
    evolver = RVEA(problem, interact=True, n_iterations=5, n_gen_per_iter=100)


    # desdeo test problem
    """
    
    problem = test_problem_builder(name="DTLZ1", n_of_variables=30, n_of_objectives=3)

    evolver = RVEA(problem, interact=True, n_iterations=5, n_gen_per_iter=400)

    plot, pref = evolver.requests()

    # show information on how to give preferences
    print(pref[0].content['message'])

    # choose to use preference method 1
    preference_method = 1  # preferred solutions
    response = np.array([1])

    # set preference information to response
    pref[preference_method - 1].response = response

    # iterate with preferences
    plot, pref = evolver.iterate(pref[preference_method - 1])

    # choose to use preference method 2
    preference_method = 2  # non-preferred solutions
    response = np.array([5])

    # set preference information to response
    pref[preference_method - 1].response = response

    # iterate with preferences
    plot, pref = evolver.iterate(pref[preference_method - 1])

    # choose to use preference method 3
    preference_method = 3  # reference point
    response = pd.DataFrame([[6.3,3.3,7,]],
                             columns=pref[preference_method - 1].content['dimensions_data'].columns)

    # set preference information to response
    pref[preference_method - 1].response = response

    # iterate with preferences
    plot, pref = evolver.iterate(pref[preference_method - 1])

    # choose to use preference method 4
    preference_method = 4  # bounds
    response = np.array([[0, 2], [1, 2], [6, 7]])

    # set preference information to response
    pref[preference_method - 1].response = response

    # iterate with preferences
    plot, pref = evolver.iterate(pref[preference_method - 1])
    """