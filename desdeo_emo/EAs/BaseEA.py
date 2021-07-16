from typing import Dict, Type, Union, Tuple

import numpy as np
import pandas as pd

from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_problem import MOProblem
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    PreferredSolutionPreference,
    NonPreferredSolutionPreference,
    BoundPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)


class eaError(Exception):
    """Raised when an error related to EA occurs"""


class BaseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__(
        self,
        a_priori: bool = False,
        interact: bool = False,
        selection_operator: Type[SelectionBase] = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
    ):
        """Initialize EA here. Set up parameters, create EA specific objects."""
        self.a_priori: bool = a_priori
        self.interact: bool = interact
        self.n_iterations: int = n_iterations
        self.n_gen_per_iter: int = n_gen_per_iter
        self.total_gen_count: int = n_gen_per_iter * n_iterations
        self.total_function_evaluations = total_function_evaluations
        self.selection_operator = selection_operator
        self.use_surrogates: bool = use_surrogates
        # Internal counters and state trackers
        self._iteration_counter: int = 0
        self._gen_count_in_curr_iteration: int = 0
        self._current_gen_count: int = 0
        self._function_evaluation_count: int = 0


    def start(self):
        """Mimics the structure of the mcdm methods. Returns the request objects from self.retuests().
        """
        return self.requests()

    def end(self):
        """To be run at the end of the evolution process.
        """
        pass

    def _next_gen(self):
        """Run one generation of an EA. Change nothing about the parameters."""

    def iterate(self, preference=None) -> Tuple:
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.
        """
        self.manage_preferences(preference)
        self._gen_count_in_curr_iteration = 0
        while self.continue_iteration():
            self._next_gen()
        self._iteration_counter += 1
        return self.requests()

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return (
            self._gen_count_in_curr_iteration < self.n_gen_per_iter
            and self.check_FE_count()
        )

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        return self._iteration_counter < self.n_iterations and self.check_FE_count()

    def check_FE_count(self) -> bool:
        """Checks whether termination criteria via function evaluation count has been
            met or not.

        Returns:
            bool: True is function evaluation count limit NOT met.
        """
        if self.total_function_evaluations == 0:
            return True
        elif self._function_evaluation_count <= self.total_function_evaluations:
            return True
        return False

    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        pass

    def requests(self) -> Tuple:
        pass


class BaseDecompositionEA(BaseEA):
    """The Base class for decomposition based EAs.

    This class contains most of the code to set up the parameters and operators.
    It also contains the logic of a simple decomposition EA.

    Parameters
    ----------
    problem : MOProblem
        The problem class object specifying the details of the problem.
    selection_operator : Type[SelectionBase], optional
        The selection operator to be used by the EA, by default None.
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
        selection_operator: Type[SelectionBase] = None,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        lattice_resolution: int = None,
        use_surrogates: bool = False,
    ):
        super().__init__(
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            selection_operator=selection_operator,
            use_surrogates=use_surrogates,
        )
        if lattice_resolution is None:
            lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
            if problem.n_of_objectives < 11:
                lattice_resolution = lattice_res_options[problem.n_of_objectives - 2]
            else:
                lattice_resolution = 3
        self.reference_vectors = ReferenceVectors(
            lattice_resolution, problem.n_of_objectives
        )
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
        # print("Using BaseDecompositionEA init")

    def _next_gen(self):
        """Run one generation of decomposition based EA. Intended to be used by
        next_iteration.
        """
        offspring = self.population.mate()  # (params=self.params)
        self.population.add(offspring, self.use_surrogates)
        selected = self._select()
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]

    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        if not isinstance(
            preference,
            (
                ReferencePointPreference,
                PreferredSolutionPreference,
                NonPreferredSolutionPreference,
                BoundPreference,
                type(None),
            ),
        ):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)}\n"
                f"{type(PreferredSolutionPreference)}\n"
                f"{type(NonPreferredSolutionPreference)}\n"
                f"{type(BoundPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)
        if preference is not None:
            if preference.request_id != self._interaction_request_id:
                msg = (
                    f"Wrong preference object sent. Expected id = "
                    f"{self._interaction_request_id}.\n"
                    f"Recieved id = {preference.request_id}"
                )
                raise eaError(msg)
        if preference is None and not self._ref_vectors_are_focused:
            self.reference_vectors.adapt(self.population.fitness)
        if isinstance(preference, ReferencePointPreference):
            ideal = self.population.ideal_fitness_val
            refpoint = (
                preference.response.values * self.population.problem._max_multiplier
            )
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            self.reference_vectors.iteractive_adapt_3(refpoint)
            self.reference_vectors.add_edge_vectors()
        elif isinstance(preference, PreferredSolutionPreference):
            self.reference_vectors.interactive_adapt_1(
                z=self.population.objectives[preference.response],
                n_solutions=np.shape(self.population.objectives)[0],
            )
            self.reference_vectors.add_edge_vectors()
        elif isinstance(preference, NonPreferredSolutionPreference):
            self.reference_vectors.interactive_adapt_2(
                z=self.population.objectives[preference.response],
                n_solutions=np.shape(self.population.objectives)[0],
            )
            self.reference_vectors.add_edge_vectors()
        elif isinstance(preference, BoundPreference):
            self.reference_vectors.interactive_adapt_4(preference.response)
            self.reference_vectors.add_edge_vectors()
        self.reference_vectors.neighbouring_angles()

    def _select(self) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population, self.reference_vectors)

    def request_plot(self) -> SimplePlotRequest:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        data = pd.DataFrame(
            self.population.objectives, columns=self.population.problem.objective_names
        )
        return SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="Objective Values"
        )

    def request_preferences(
        self
    ) -> Union[
        None,
        Tuple[
            PreferredSolutionPreference,
            NonPreferredSolutionPreference,
            ReferencePointPreference,
            BoundPreference,
        ],
    ]:

        if self.a_priori is False and self.interact is False:
            return
        if (
            self.a_priori is True
            and self.interact is False
            and self._iteration_counter > 0
        ):
            return
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        message = (
            "Please provide preferences. There is four ways to do this. You can either:\n\n"
            "\t1: Select preferred solution(s)\n"
            "\t2: Select non-preferred solution(s)\n"
            "\t3: Specify a reference point worse than or equal to the ideal point\n"
            "\t4: Specify desired ranges for objectives.\n\n"
            "In case you choose \n\n"
            "1, please specify index/indices of preferred solutions in a numpy array (indexing starts from 0).\n"
            "For example: \n"
            "\tnumpy.array([1]), for choosing the solutions with index 1.\n"
            "\tnumpy.array([2, 4, 5, 16]), for choosing the solutions with indices 2, 4, 5, and 16.\n\n"
            "2, please specify index/indices of non-preferred solutions in a numpy array (indexing starts from 0).\n"
            "For example: \n"
            "\tnumpy.array([3]), for choosing the solutions with index 3.\n"
            "\tnumpy.array([1, 2]), for choosing the solutions with indices 1 and 2.\n\n"
            "3, please provide a reference point worse than or equal to the ideal point:\n\n"
            f"{dimensions_data.loc['ideal']}\n"
            f"The reference point will be used to focus the reference vectors towards "
            f"the preferred region.\n"
            f"If a reference point is not provided, the previous state of the reference"
            f" vectors is used.\n"
            f"If the reference point is the same as the ideal point, "
            f"the reference vectors are spread uniformly in the objective space.\n\n"
            "4, please specify desired lower and upper bound for each objective, starting from \n"
            "the first objective and ending with the last one. Please specify the bounds as a numpy array containing \n"
            "lists, so that the first item of list is the lower bound and the second the upper bound, for each \n"
            "objective. \n"
            "\tFor example: numpy.array([[1, 2], [2, 5], [0, 3.5]]), for problem with three "
            "objectives.\n"
            f"Ideal vector: \n{dimensions_data.loc['ideal']}\n"
            f"Nadir vector: \n{dimensions_data.loc['nadir']}."
        )

        def validator(dimensions_data: pd.DataFrame, reference_point: pd.DataFrame):
            validate_ref_point_dimensions(dimensions_data, reference_point)
            validate_ref_point_data_type(reference_point)
            validate_ref_point_with_ideal(dimensions_data, reference_point)
            return

        interaction_priority = "recommended"
        self._interaction_request_id = np.random.randint(0, 1e9)

        # return multiple preference-requests, user decides with request will (s)he respond to by using an index.
        return (
            PreferredSolutionPreference(
                n_solutions=self.population.objectives.shape[0],
                message=message,
                interaction_priority=interaction_priority,
                request_id=self._interaction_request_id,
            ),
            NonPreferredSolutionPreference(
                n_solutions=self.population.objectives.shape[0],
                message=None,
                interaction_priority=interaction_priority,
                request_id=self._interaction_request_id,
            ),
            ReferencePointPreference(
                dimensions_data=dimensions_data,
                message=None,
                interaction_priority=interaction_priority,
                preference_validator=validator,
                request_id=self._interaction_request_id,
            ),
            BoundPreference(
                dimensions_data=dimensions_data,
                n_objectives=self.population.problem.n_of_objectives,
                message=None,
                interaction_priority=interaction_priority,
                request_id=self._interaction_request_id,
            ),
        )

    def requests(self) -> Tuple:
        return (self.request_preferences(), self.request_plot())


    def end(self):
        """Conducts non-dominated sorting at the end of the evolution process
        Returns:
            tuple: The first element is a 2-D array of the decision vectors of the non-dominated solutions.
                The second element is a 2-D array of the corresponding objective values.
        """
        non_dom = self.population.non_dominated_objectives()
        return (
            self.population.individuals[non_dom, :],
            self.population.objectives[non_dom, :],
        )
