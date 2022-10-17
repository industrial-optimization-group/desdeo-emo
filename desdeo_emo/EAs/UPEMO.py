from typing import Dict, List, Union

import numpy as np
import pandas as pd
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, BaseEA, eaError
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.IOPIS_APD import IOPIS_APD_Select
from desdeo_emo.selection.IOPIS_NSGAIII import IOPIS_NSGAIII_select
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_problem import MOProblem
from desdeo_tools.interaction import (
    BoundPreference,
    NonPreferredSolutionPreference,
    PreferredSolutionPreference,
    ReferencePointPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)
from scipy.special import comb
from desdeo_emo.selection.RNSGAIII_select import RNSGAIII_select


class UPEMO(BaseDecompositionEA, BaseEA):
    def __init__(
        self,
        problem: MOProblem,
        population_size: int,
        n_survive: int = None,
        selection_type: str = None,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        a_priori: bool = True
        interact: bool = True
        BaseEA.__init__(
            self=self,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )

        self.n_ref_points = 0
        #self.pop_size_rp = population_size_per_rp
        self.ref_points = None
        temp_lattice_resolution = 0
        temp_number_of_vectors = 0

        while True:
            temp_lattice_resolution += 1
            temp_number_of_vectors = comb(
                temp_lattice_resolution + problem.n_of_objectives - 1,
                problem.n_of_objectives - 1,
                exact=True,
            )
            if temp_number_of_vectors > population_size:
                break

        reference_vectors = ReferenceVectors(
            lattice_resolution=temp_lattice_resolution,
            number_of_objectives=problem.n_of_objectives,
        )

        self.problem = problem
        self.lattice_resolution = temp_lattice_resolution - 1
        self.population_size = population_size
        self.population = None

        #pop_size = (temp_number_of_vectors * self.n_ref_points) + problem.n_of_objectives
    
        self.selection_type = selection_type
        #selection_operator = RNSGAIII_select(
        #    self.population, self.pop_size_rp, self.ref_points, n_survive, selection_type=selection_type,        )
        self.selection_operator = None
        #self._interaction_location = "Selection"

    def iterate(self, preference=None) -> Tuple:
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.
        """
        self.manage_preferences(preference)
        evolver = RNSGAIII(problem, 50, ref_points, n_iterations=10, n_gen_per_iter=30)
        while evolver.continue_evolution():
            pref, plots = evolver.iterate()
            print(f"Running iteration {evolver._iteration_counter}")
        
        #self.pre_iteration()
        #self._gen_count_in_curr_iteration = 0
        #while self.continue_iteration():
        #    self._next_gen()
        #self._iteration_counter += 1
        self.post_iteration()
        return self.requests()

    def manage_preferences(
        self,
        pop: Population,
        preference: Union[
            PreferredSolutionPreference,
            NonPreferredSolutionPreference,
            ReferencePointPreference,
            BoundPreference,
            None,
        ],
    ):

        if preference is None:
            msg = "Giving preferences is mandatory"
            raise eaError(msg)
        if self._interaction_request_id != preference.request_id:
            raise ValueError("Wrong request object provided. Request IDs don't match.")
        if self.interaction_type == "Reference point":
            return self.manage_reference_point(pop, preference)
        if self.interaction_type == "Preferred solutions":
            return self.manage_preferred_solutions(pop, preference)
        if self.interaction_type == "Non-preferred solutions":
            return self.manage_non_preferred_solutions(pop, preference)
        if self.interaction_type == "Preferred ranges":
            return self.manage_preferred_ranges(preference)
        raise ValueError("Interaction type not set.")


    def request_preferences(
        self, pop: Population
    ) -> Union[
        PreferredSolutionPreference,
        NonPreferredSolutionPreference,
        ReferencePointPreference,
        BoundPreference,
    ]:
        if self.interaction_type == "Reference point":
            return self.request_reference_point(pop)
        if self.interaction_type == "Preferred solutions":
            return self.request_preferred_solutions(pop)
        if self.interaction_type == "Non-preferred solutions":
            return self.request_non_preferred_solutions(pop)
        if self.interaction_type == "Preferred ranges":
            return self.request_preferred_ranges(pop)
        raise ValueError("Interaction type not set.")


    def request_reference_point(self, pop: Population) -> ReferencePointPreference:
        dimensions_data = pd.DataFrame(
        index=["minimize", "ideal", "nadir"],
        columns=pop.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = pop.problem._max_multiplier
        dimensions_data.loc["ideal"] = pop.ideal_objective_vector
        dimensions_data.loc["nadir"] = pop.nadir_objective_vector
        message = (
            f"Please provide a reference point worse than the ideal point:\n\n"
            f"{dimensions_data.loc['ideal']}\n"
            f"The reference point will be used to focus the reference vectors towards "
            f"the preferred region.\n"
            f"If a reference point is not provided, the reference vectors are spread uniformly"
            f" in the objective space."
        )

        def validator(dimensions_data: pd.DataFrame, reference_point: pd.DataFrame):
            validate_ref_point_dimensions(dimensions_data, reference_point)
            validate_ref_point_data_type(reference_point)
            validate_ref_point_with_ideal(dimensions_data, reference_point)
            return

        self._interaction_request_id = np.random.randint(0, 1e9)

        return ReferencePointPreference(
            dimensions_data=dimensions_data,
            message=message,
            interaction_priority="recommended",
            preference_validator=validator,
            request_id=self._interaction_request_id,
        )

    def request_preferred_solutions(
        self, pop: Population
    ) -> PreferredSolutionPreference:
        message = (
            "Please specify index/indices of preferred solutions in a numpy array (indexing starts from 0).\n"
            "For example: \n"
            "\tnumpy.array([1]), for choosing the solutions with index 1.\n"
            "\tnumpy.array([2, 4, 5, 16]), for choosing the solutions with indices 2, 4, 5, and 16.\n"
            "The reference vectors will be focused around the chosen preferred solutions."
        )

        self._interaction_request_id = np.random.randint(0, 1e9)

        return PreferredSolutionPreference(
            n_solutions=pop.objectives.shape[0],
            message=message,
            interaction_priority="recommended",
            request_id=self._interaction_request_id,
        )

    def request_non_preferred_solutions(
        self, pop: Population
    ) -> NonPreferredSolutionPreference:

        message = (
            "Please specify index/indices of non-preferred solutions in a numpy array (indexing starts from 0).\n"
            "For example: \n"
            "\tnumpy.array([3]), for choosing the solutions with index 3.\n"
            "\tnumpy.array([1, 2]), for choosing the solutions with indices 1 and 2.\n"
            "The reference vectors are focused away from the non*åreferred solutions."
        )

        self._interaction_request_id = np.random.randint(0, 1e9)
        return NonPreferredSolutionPreference(
            n_solutions=pop.objectives.shape[0],
            message=message,
            interaction_priority="recommended",
            request_id=self._interaction_request_id,
        )

    def request_preferred_ranges(self, pop) -> BoundPreference:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=pop.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = pop.problem._max_multiplier
        dimensions_data.loc["ideal"] = pop.ideal_objective_vector
        dimensions_data.loc["nadir"] = pop.nadir_objective_vector
        message = (
            "Please specify desired lower and upper bound for each objective, starting from \n"
            "the first objective and ending with the last one. Please specify the bounds as a numpy array containing \n"
            "lists, so that the first item of list is the lower bound and the second the upper bound, for each \n"
            "objective. \n"
            "The reference vectors are regenerated within the preferred ranges.\n"
            "\tFor example: numpy.array([[1, 2], [2, 5], [0, 3.5]]), for a problem with three "
            "objectives.\n"
            f"Ideal vector: \n{dimensions_data.loc['ideal']}\n"
            f"Nadir vector: \n{dimensions_data.loc['nadir']}."
        )
        self._interaction_request_id = np.random.randint(0, 1e9)

        BoundPreference(
            dimensions_data=dimensions_data,
            n_objectives=pop.problem.n_of_objectives,
            message=message,
            interaction_priority="recommended",
            request_id=self._interaction_request_id,
        )

    def manage_reference_point(
        self, pop: Population, preference: ReferencePointPreference
    ):
        if not isinstance(preference, ReferencePointPreference):
            raise TypeError(
                "Preference object must be an instance of ReferencePointPreference."
            )
        ideal = pop.ideal_fitness_val
        refpoint = preference.response.values * pop.problem._max_multiplier
        refpoint = refpoint - ideal
        norm = np.sqrt(np.sum(np.square(refpoint)))
        refpoint = refpoint / norm
        self.vectors.iteractive_adapt_3(refpoint)
        self.vectors.add_edge_vectors()
        self.vectors.neighbouring_angles()

    def manage_preferred_solutions(
        self, pop: Population, preference: PreferredSolutionPreference
    ):
        if not isinstance(preference, PreferredSolutionPreference):
            raise TypeError(
                "Preference object must be an instance of PreferredSolutionPreference."
            )
        self.vectors.interactive_adapt_1(
            z=pop.objectives[preference.response],
            n_solutions=np.shape(pop.objectives)[0],
        )
        self.vectors.add_edge_vectors()
        self.vectors.neighbouring_angles()

    def manage_non_preferred_solutions(
        self, pop: Population, preference: NonPreferredSolutionPreference
    ):
        if not isinstance(preference, NonPreferredSolutionPreference):
            raise TypeError(
                "Preference object must be an instance of NonPreferredSolutionPreference."
            )
        self.vectors.interactive_adapt_2(
            z=pop.objectives[preference.response],
            n_solutions=np.shape(pop.objectives)[0],
        )
        self.vectors.add_edge_vectors()
        self.vectors.neighbouring_angles()

    def manage_preferred_ranges(self, preference: BoundPreference):
        if not isinstance(preference, BoundPreference):
            raise TypeError("Preference object must be an instance of BoundPreference.")
        self.vectors.interactive_adapt_4(preference.response)
        self.vectors.add_edge_vectors()
        self.vectors.neighbouring_angles()

    def _select(self) -> List:
        return self.selection_operator.do(
            self.population, self.reference_vectors, self._preference
        )



