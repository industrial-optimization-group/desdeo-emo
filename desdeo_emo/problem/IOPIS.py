from typing import Dict, List, Union

import numpy as np
import pandas as pd
from desdeo_emo.population.Population import Population
from desdeo_problem import (
    MOProblem,
    ScalarConstraint,
    ScalarObjective,
    Variable,
    VectorObjective,
)
from desdeo_tools.maps.preference_incorporated_space_RP import (
    IOPIS,
    MultiDMPIS,
    classificationPIS,
)
from desdeo_tools.scalarization import AUG_GUESS_GLIDE, AUG_STOM_GLIDE


class IOPISProblem(MOProblem):
    """A problem class for the IOPIS formulation for interactive optimization.

    This variant uses the classification kind of preference information for
    the creation of the Preference incorporated space (PIS).

    Arguments:
        objectives (List[Union[ScalarObjective, VectorObjective]]): A list containing
            the objectives of the problem.
        variables (List[Variable]): A list containing the variables of the problem.
        nadir (np.ndarray): Nadir point of the problem.
        ideal (np.ndarray): Ideal point of the problem.
        PIS: An instantiated classificationPIS class from desdeo-tools.
        constraints (List[ScalarConstraint], optional): A list of the constraints of the problem. Defaults to None.
    """

    def __init__(
        self,
        objectives: List[Union[ScalarObjective, VectorObjective]],
        variables: List[Variable],
        nadir: np.ndarray,
        ideal: np.ndarray,
        PIS_type: str,
        initial_preference: Union[List, Dict],
        constraints: List[ScalarConstraint] = None,
    ):
        super().__init__(
            objectives=objectives,
            variables=variables,
            constraints=constraints,
            nadir=nadir,
            ideal=ideal,
        )
        allowed_PIS_types = ["MultiDM", "Classification", "IOPIS"]
        self.PIS_type = PIS_type
        if PIS_type not in allowed_PIS_types:
            raise ValueError(f"PIS_type should be one of {allowed_PIS_types}")
        if PIS_type == "IOPIS":
            self.PIS = IOPIS(
                utopian=self.ideal * self._max_multiplier,
                nadir=self.nadir * self._max_multiplier,
                scalarizers=[AUG_STOM_GLIDE, AUG_GUESS_GLIDE],
            )
            self.fitness_names = ["STOM Value", "GUESS Value"]
            self.allowable_interaction_types = {
                "Reference point": (
                    "Specify a reference point worse than the utopian point. ",
                    "The STOM and GUESS scalarization functions are used to find "
                    "solutions that approximate the given reference point. "
                    "This results in solutions that represent the trade-offs between "
                    "the optimal solution for STOM and GUESS."
                    " The closer the reference point is to the Pareto front, the smaller "
                    "the spread of the solutions is.",
                )
            }
        elif PIS_type == "Classification":
            self.PIS = classificationPIS(
                utopian=self.ideal * self._max_multiplier,
                nadir=self.nadir * self._max_multiplier,
                scalarizers=[AUG_STOM_GLIDE],
            )
            self.fitness_names = ["NIMBUS Value", "STOM Value"]
            self.allowable_interaction_types = {
                "Classification": (
                    "Classify the objectives into 'make better', 'keep same', "
                    "'allow to worsen'."
                )
            }
        elif PIS_type == "MultiDM":
            self.PIS = MultiDMPIS(
                utopian=self.ideal * self._max_multiplier,
                nadir=self.nadir * self._max_multiplier,
                num_DM=2,
            )
            self.fitness_names = [
                "First preference fitness",
                "Second preference fitness",
            ]
            self.allowable_interaction_types = {
                "Reference points": (
                    "Specify two reference points worse than the utopian point. ",
                    "The STOM scalarization functions are used to find "
                    "solutions that approximate the given reference points. "
                    "This results in solutions that represent the trade-offs between "
                    "the optimal solution for the two reference points."
                    " The closer the reference points are to each other, the smaller "
                    "the spread of the solutions is.",
                )
            }
        self.PIS.update_preference(initial_preference)
        self.ideal_fitness = self.PIS(self.ideal * self._max_multiplier)
        self.nadir_fitness = self.PIS(self.nadir * self._max_multiplier)

        self.num_dim_fitness = self.PIS.num_scalarizers

    @property
    def n_of_fitnesses(self) -> int:
        return self.num_dim_fitness

    def evaluate_fitness(self, objective_vectors: np.ndarray) -> np.ndarray:
        """Evaluate objective fitness.

        Arguments:
            objective_vectors (np.ndarray): objective vectors

        Returns:
            np.ndarray: Objective fitness
        """
        return self.PIS(objective_vectors * self._max_multiplier)

    def reevaluate_fitness(self, objective_vectors: np.ndarray) -> np.ndarray:
        """Re-evaluate objective fitness.

        Calls update_ideal with objective_vectors.

        Arguments:
            objective_vectors (np.ndarray): objective vectors

        Returns:
            np.ndarray: Objective fitness
        """
        fitness = self.PIS(objective_vectors * self._max_multiplier)
        self.ideal_fitness = self.PIS(self.ideal * self._max_multiplier)
        self.update_ideal(objective_vectors, fitness)
        return fitness

    def update_preference(self, preference: Union[Dict, List]):
        """Update PIS preference

        Arguments:
            preference (Dict): PIS preferences

        """
        self.PIS.update_preference(preference)

    def update_ideal(self, objective_vectors: np.ndarray, fitness: np.ndarray):
        """Update ideal vector.

        Arguments:
            objective_vectors (np.ndarray): Objective vectors
            fitness (np.ndarray): Fitness values for objective vectors

        """
        self.ideal_fitness = np.amin(np.vstack((self.ideal_fitness, fitness)), axis=0)

        self.ideal = (
            np.amin(
                np.vstack((self.ideal, objective_vectors)) * self._max_multiplier,
                axis=0,
            )
            * self._max_multiplier
        )
        self.PIS.update_map(
            utopian=self.ideal * self._max_multiplier,
            nadir=self.nadir * self._max_multiplier,
        )

    def request_preferences(self, pop):
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=pop.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = pop.problem._max_multiplier
        dimensions_data.loc["ideal"] = pop.ideal_objective_vector
        dimensions_data.loc["nadir"] = pop.nadir_objective_vector
        message = (
            f"Provide the following preference type: \n"
            f"{self.allowable_interaction_types}"
        )
        return dimensions_data, message

    def manage_preferences(self, population: Population, preference=None):
        if self.PIS_type == "IOPIS":
            if not isinstance(preference, dict):
                raise TypeError("Preference type must be dict")
            self.PIS.update_map(
                utopian=population.ideal_objective_vector
                * population.problem._max_multiplier,
                nadir=population.nadir_objective_vector
                * population.problem._max_multiplier,
            )
            self.PIS.update_preference(preference)
            return
        elif self.PIS_type == "MultiDM":
            if not isinstance(preference, list):
                raise TypeError("Preference type must be list")

            self.PIS.update_map(
                utopian=population.ideal_objective_vector
                * population.problem._max_multiplier,
                nadir=population.nadir_objective_vector
                * population.problem._max_multiplier,
            )
            self.PIS.update_preference(preference)
            return
        else:
            raise TypeError("Classification not implemented in this version yet.")
