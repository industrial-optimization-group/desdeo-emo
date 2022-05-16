from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_tools.interaction import (
    BoundPreference,
    NonPreferredSolutionPreference,
    PreferredSolutionPreference,
    ReferencePointPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)


class SelectionBase(ABC):
    """The base class for the selection operator.
    """

    @abstractmethod
    def do(self, pop: Population, *args) -> List[int]:
        """Use the selection operator over the given fitness values. Return the indices
            individuals with the best fitness values according to the operator.

        Parameters
        ----------
        fitness : np.ndarray
            Fitness of the individuals from which the next generation is to be selected.

        Returns
        -------
        List[int]
            The list of selected individuals
        """


class InteractiveDecompositionSelectionBase(SelectionBase):
    """The base class for the selection operators for interactive decomposition based MOEAs.
    """

    def __init__(
        self,
        number_of_vectors: int,
        number_of_objectives: int,
        selection_type: str = None,
    ):
        self.vectors = ReferenceVectors(
            number_of_vectors=number_of_vectors,
            number_of_objectives=number_of_objectives,
        )
        if selection_type is None:
            selection_type = "mean"
        self.selection_type = selection_type
        self._interaction_request_id: int = None
        self.allowable_interaction_types = {
            "Reference point": (
                "Specify a reference point worse than the utopian point. "
                "The reference vectors are focused around the vector joining "
                "provided reference point and the utopian point. "
                "New solutions are searched for in this focused region of interest."
            ),
            "Preferred solutions": (
                "Choose one or more solutions as the preferred solutions. "
                "The reference vectors are focused around the vector joining "
                "the utopian point and the preferred solutions. "
                "New solutions are searched for in this focused regions of interest."
            ),
            "Non-preferred solutions": (
                "Choose one or more solutions that are not preferred. "
                "The reference vectors near such solutions are removed. "
                "New solutions are hence not searched for in areas close to these solutions."
            ),
            "Preferred ranges": (
                "Provide preferred values for the upper and lower bounds of all objectives. "
                "New reference vectors are generated within these bounds. "
                "New solutions are searched for in this bounded region of interest."
            ),
        }
        self.interaction_type: str = None

    def _calculate_fitness(self, pop: Population) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity

    def set_interaction_type(self, interaction_type: str = None) -> Union[None, dict]:
        if interaction_type is None:
            return self.allowable_interaction_types
        elif interaction_type not in self.allowable_interaction_types.keys():
            raise ValueError(
                f"Invalid interaction type. Interaction type must be one of {self.allowable_interaction_types.keys()}."
            )
        else:
            self.interaction_type = interaction_type

    def adapt_RVs(self, fitness: np.ndarray) -> None:
        self.vectors.adapt(fitness)
        self.vectors.neighbouring_angles()

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
            return self.adapt_RVs(pop.fitness)
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
