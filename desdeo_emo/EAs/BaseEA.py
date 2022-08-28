from typing import Dict, Literal, Tuple, Type, Union

import pandas as pd
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase

from desdeo_problem import MOProblem, DataProblem, EvaluationResults
from desdeo_emo.problem import IOPISProblem
from desdeo_tools.interaction import SimplePlotRequest
from desdeo_tools.interaction.request import BaseRequest
from scipy.special import comb
from desdeo_tools.utilities import non_dominated
from desdeo_emo.utilities.non_dominated import check_domination


class eaError(Exception):
    """Raised when an error related to EA occurs"""


class BaseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__(
        self,
        a_priori: bool = False,
        interact: bool = False,
        selection_operator: Type[
            SelectionBase
        ] = None,  # TODO: No algorithm uses this option. Remove?
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        keep_archive: bool = False,
        save_non_dominated: bool = False,
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
        self._interaction_location: Union[
            None, Literal["Population", "Selection", "EA", "Problem"]
        ] = None
        self.interaction_type_set_bool: bool = False
        self.allowable_interaction_types: Union[None, Dict] = None
        self.population: Union[None, Population] = None
        self.keep_archive = keep_archive
        self.archive: dict = {}
        self.save_non_dominated = save_non_dominated

    """@property
    def interaction_location(
        self
    ) -> Union[None, Literal["Population", "Selection", "EA", "Problem"]]:
        ""Return the location of interaction handling
        ""
        return self._interaction_location"""

    @property
    def allowable_interaction_types(self) -> dict:
        if self._allowable_interaction_types is not None:
            return self._allowable_interaction_types
        else:
            self.allowable_interaction_types = self.set_interaction_type(None)
            return self._allowable_interaction_types

    @allowable_interaction_types.setter
    def allowable_interaction_types(self, value: dict):
        self._allowable_interaction_types = value

    def start(self):
        """Mimics the structure of the mcdm methods. Returns the request objects from self.retuests()."""
        if self.population is None:
            raise eaError("Population not initialized.")
        if self.selection_operator is None:
            raise eaError("Selection operator not initialized.")
        if self.interact and not self.interaction_type_set_bool:
            raise eaError(
                "Interaction type not set. Use the set_interaction_type() method."
            )
        if self.save_non_dominated:
            self.non_dominated = {}
            non_dom_indices = non_dominated(self.population.fitness)
            self.non_dominated["individuals"] = self.population.individuals[
                non_dom_indices
            ]
            self.non_dominated["objectives"] = self.population.objectives[
                non_dom_indices
            ]
            self.non_dominated["fitness"] = self.population.fitness[non_dom_indices]
            # TODO: add uncertainity, constraints, and generation numbers
        return self.requests()

    def end(self):
        """To be run at the end of the evolution process."""
        if self.keep_archive:
            objective_names = self.population.problem.objective_names
            if isinstance(objective_names[0], list):
                objective_names = [
                    item for sublist in objective_names for item in sublist
                ]
            fitness_names = self.population.problem.fitness_names
            if isinstance(fitness_names[0], list):
                objective_names = [
                    item for sublist in fitness_names for item in sublist
                ]
            self.archive[self._current_gen_count] = {
                "iteration": self._iteration_counter,
                "decision variables": pd.DataFrame(
                    self.population.individuals,
                    columns=self.population.problem.variable_names,
                ),
                "objectives": pd.DataFrame(
                    self.population.objectives,
                    columns=objective_names,
                ),
                "fitness": pd.DataFrame(
                    self.population.fitness,
                    columns=fitness_names,
                ),
            }
            return self.archive

    def _next_gen(self):
        """Run one generation of an EA. Change nothing about the parameters."""

    def iterate(self, preference=None) -> Tuple:
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.
        """
        self.manage_preferences(preference)
        self.pre_iteration()
        self._gen_count_in_curr_iteration = 0
        while self.continue_iteration():
            self._next_gen()
        self._iteration_counter += 1
        self.post_iteration()
        return self.requests()

    def pre_iteration(self):
        """Run this code before every iteration."""
        if self.keep_archive:
            objective_names = self.population.problem.objective_names
            if isinstance(objective_names[0], list):
                objective_names = [
                    item for sublist in objective_names for item in sublist
                ]
            fitness_names = self.population.problem.fitness_names
            if isinstance(fitness_names[0], list):
                objective_names = [
                    item for sublist in fitness_names for item in sublist
                ]
            self.archive[self._current_gen_count] = {
                "iteration": self._iteration_counter,
                "decision variables": pd.DataFrame(
                    self.population.individuals,
                    columns=self.population.problem.variable_names,
                ),
                "objectives": pd.DataFrame(
                    self.population.objectives,
                    columns=objective_names,
                ),
                "fitness": pd.DataFrame(
                    self.population.fitness,
                    columns=fitness_names,
                ),
            }

    def post_iteration(self):
        """Run this code after every iteration."""
        return

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

    def set_interaction_type(
        self, interaction_type: Union[None, str]
    ) -> Union[None, str]:
        if self._interaction_location == "Selection":
            try:
                self.interaction_type_set_bool = True
                return self.selection_operator.set_interaction_type(interaction_type)
            except NotImplementedError as e:
                self.interaction_type_set_bool = False
                raise eaError(
                    "Interaction not implemented in the selection operator."
                ) from e
        if self._interaction_location == "Problem":
            try:
                self.interaction_type_set_bool = True
                return
            except NotImplementedError as e:
                self.interaction_type_set_bool = False
                raise eaError(
                    "Interaction not implemented in the problem class."
                ) from e
        if self._interaction_location == "EA":
            self.interaction_type_set_bool = False
            raise NotImplementedError("No interaction type implemented yet.")
        if self._interaction_location == "Population":
            try:
                self.interaction_type_set_bool = True
                return self.selection_operator.set_interaction_type(interaction_type)
            except NotImplementedError as e:
                self.interaction_type_set_bool = False
                raise eaError(
                    "Interaction not implemented in the population class."
                ) from e

    def manage_preferences(self, preference=None):
        """Forward the preference to the correct preference handling method.

        Args:
            preference (_type_, optional): _description_. Defaults to None.

        Raises:
            eaError: Preference handling not implemented.
        """
        if self._interaction_location is None:
            return
        if self._interaction_location == "Population":
            try:
                self.population.manage_preferences(preference)
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in population."
                ) from e
        if self._interaction_location == "Selection":
            try:
                self.selection_operator.manage_preferences(self.population, preference)
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in selection."
                ) from e
        if self._interaction_location == "EA":
            raise eaError("Override this method in child class.")
        if self._interaction_location == "Problem":
            try:
                self.population.problem.manage_preferences(self.population, preference)
                self.population.reevaluate_fitness()
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in problem class."
                ) from e

    def requests(self) -> Tuple:
        pass

    def non_dominated_archive(
        self, individuals: np.ndarray, results: EvaluationResults
    ) -> None:
        assert self.save_non_dominated
        new_fitness = results.fitness
        new_objectives = results.objectives
        if new_fitness.ndim == 1:
            new_fitness = np.atleast_2d(new_fitness)
            new_objectives = np.atleast_2d(results.objectives)
        if individuals.ndim == 1:
            individuals = np.atleast_2d(individuals)
        new_indices, old_indices = check_domination(
            new_fitness, self.non_dominated["fitness"]
        )
        self.non_dominated["individuals"] = np.vstack(
            (self.non_dominated["individuals"][old_indices], individuals[new_indices])
        )
        self.non_dominated["objectives"] = np.vstack(
            (
                self.non_dominated["objectives"][old_indices],
                new_objectives[new_indices],
            )
        )
        self.non_dominated["fitness"] = np.vstack(
            (self.non_dominated["fitness"][old_indices], new_fitness[new_indices])
        )


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
        problem: MOProblem = None,
        initial_population: Population = None,
        population_size: int = None,
        population_params: Dict = None,
        lattice_resolution: int = None,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        use_surrogates: bool = False,
        total_function_evaluations: int = 0,
        keep_archive: bool = False,
        save_non_dominated: bool = False,
    ):
        super().__init__(
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
            keep_archive=keep_archive,
            save_non_dominated=save_non_dominated,
        )
        if interact:
            if isinstance(problem, (MOProblem, DataProblem)):
                self._interaction_location = "Selection"
            if isinstance(problem, (IOPISProblem)):
                self._interaction_location = "Problem"
                self.set_interaction_type(None)
        if initial_population is not None:
            if problem is not None:
                raise eaError("Provide only one of initial_population or problem.")
            #  Population should be compatible.
            self.population = initial_population  # TODO put checks here.
            num_fitnesses = self.population.problem.n_of_fitnesses
            if population_size is None and lattice_resolution is None:
                population_size = self.population.pop_size
        else:
            if problem is None:
                raise eaError("Provide one of initial_population or problem.")
            num_fitnesses = problem.n_of_fitnesses
            if population_size is None:
                if lattice_resolution is None:
                    lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
                    if num_fitnesses < 11:
                        lattice_resolution = lattice_res_options[num_fitnesses - 2]
                    else:
                        lattice_resolution = 3
                population_size = comb(
                    lattice_resolution + num_fitnesses - 1,
                    num_fitnesses - 1,
                    exact=True,
                )

            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size
        # self.reference_vectors = ReferenceVectors(lattice_resolution, num_fitnesses)
        # print("Using BaseDecompositionEA init")

    def _next_gen(self):
        """Run one generation of decomposition based EA. Intended to be used by
        next_iteration.
        """
        offspring = self.population.mate()  # (params=self.params)
        if self.save_non_dominated:
            results = self.population.add(offspring, self.use_surrogates)
            self.non_dominated_archive(offspring, results)
        else:
            self.population.add(offspring, self.use_surrogates)
        selected = self._select()
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        if not self.use_surrogates:
            self._function_evaluation_count += offspring.shape[0]

    def pre_iteration(self):
        super().pre_iteration()
        if not self.interact:
            self.selection_operator.adapt_RVs(self.population.fitness)
        if self._interaction_location != "Selection":
            self.selection_operator.adapt_RVs(self.population.fitness)

    def _select(self) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population)

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

    def request_preferences(self) -> Type[BaseRequest]:

        if self.interact is False:
            return
        if self._interaction_location == "Problem":
            return self.population.problem.request_preferences(self.population)
        if self._interaction_location == "Selection":
            return self.selection_operator.request_preferences(self.population)

    def requests(self) -> Tuple:
        return (self.request_preferences(), self.request_plot())

    def end(self):
        """Conducts non-dominated sorting at the end of the evolution process

        Returns:
            tuple: The first element is a 2-D array of the decision vectors of the non-dominated solutions.
                The second element is a 2-D array of the corresponding objective values.
        """
        base_return = super().end()
        non_dom = self.population.non_dominated_objectives()
        return (
            self.population.individuals[non_dom, :],
            self.population.objectives[non_dom, :],
            base_return,
        )
