from typing import Dict, Union, List

from desdeo_emo.EAs.BaseEA import eaError
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, BaseEA
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.IOPIS_APD import IOPIS_APD_Select
from desdeo_emo.selection.IOPIS_NSGAIII import IOPIS_NSGAIII_select
from desdeo_problem.Problem import MOProblem
from desdeo_tools.scalarization import StomASF, PointMethodASF, AugmentedGuessASF
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_tools.interaction import (
    ReferencePointPreference,
    validate_ref_point_with_ideal_and_nadir,
)
import numpy as np
import pandas as pd


class BaseIOPISDecompositionEA(BaseDecompositionEA, BaseEA):
    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
    ):
        a_priori: bool = True
        interact: bool = True
        if problem.ideal is None or problem.nadir is None:
            msg = (
                f"The problem instance should contain the information about ideal and "
                f"nadir point."
            )
            raise eaError(msg)

        BaseEA.__init__(
            self=self,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )

        scalarization_methods = [
            StomASF(ideal=problem.ideal * problem._max_multiplier),
            # PointMethodASF(
            #     nadir=problem.nadir * problem._max_multiplier,
            #     ideal=problem.ideal * problem._max_multiplier,
            # ),
            AugmentedGuessASF(
                nadir=problem.nadir * problem._max_multiplier,
                ideal=problem.ideal * problem._max_multiplier,
                indx_to_exclude=[],
            ),
        ]
        if lattice_resolution is None:
            lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
            if len(scalarization_methods) < 11:
                lattice_resolution = lattice_res_options[len(scalarization_methods) - 2]
            else:
                lattice_resolution = 3
        reference_vectors = ReferenceVectors(
            lattice_resolution=lattice_resolution,
            number_of_objectives=len(scalarization_methods),
        )
        population_size = reference_vectors.number_of_vectors
        population = Population(problem, population_size, population_params)

        self.reference_vectors = reference_vectors
        self.scalarization_methods = scalarization_methods

        if initial_population is not None:
            #  Population should be compatible.
            self.population = initial_population  # TODO put checks here.
        elif initial_population is None:
            if population_size is None:
                population_size = self.reference_vectors.number_of_vectors
            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size
        self._ref_vectors_are_focused: bool = False

    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        if preference is None:
            msg = "Giving preferences is mandatory"
            raise eaError(msg)

        if not isinstance(preference, ReferencePointPreference):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)

        if preference.request_id != self._interaction_request_id:
            msg = (
                f"Wrong preference object sent. Expected id = "
                f"{self._interaction_request_id}.\n"
                f"Recieved id = {preference.request_id}"
            )
            raise eaError(msg)

        refpoint = preference.response.values * self.population.problem._max_multiplier
        self._preference = refpoint
        scalarized_space_fitness = np.asarray(
            [
                scalar(self.population.fitness, self._preference)
                for scalar in self.scalarization_methods
            ]
        ).T
        self.reference_vectors.adapt(scalarized_space_fitness)
        self.reference_vectors.neighbouring_angles()

    def request_preferences(self) -> Union[None, ReferencePointPreference]:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        message = (
            f"Provide a reference point worse than to the ideal point and better than"
            f" the nadir point.\n"
            f"Ideal point: \n{dimensions_data.loc['ideal']}\n"
            f"Nadir point: \n{dimensions_data.loc['nadir']}\n"
            f"The reference point will be used to create scalarization functions in "
            f"the preferred region.\n"
        )
        interaction_priority = "required"
        self._interaction_request_id = np.random.randint(0, 1e7)
        return ReferencePointPreference(
            dimensions_data=dimensions_data,
            message=message,
            interaction_priority=interaction_priority,
            preference_validator=validate_ref_point_with_ideal_and_nadir,
            request_id=self._interaction_request_id,
        )

    def _select(self) -> List:
        return self.selection_operator.do(
            self.population, self.reference_vectors, self._preference
        )


class IOPIS_RVEA(BaseIOPISDecompositionEA, RVEA):
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
        alpha: float = None,
        lattice_resolution: int = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        time_penalty_component: Union[str, float] = None,
        use_surrogates: bool = False,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )
        self.time_penalty_component = time_penalty_component
        time_penalty_component_options = ["original", "function_count", "interactive"]
        if time_penalty_component is None:
            if self.interact is True:
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
        selection_operator = IOPIS_APD_Select(
            self.time_penalty_function, self.scalarization_methods, self.alpha
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


class IOPIS_NSGAIII(BaseIOPISDecompositionEA):
    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )
        self.selection_operator = IOPIS_NSGAIII_select(
            self.scalarization_methods, self.population
        )

