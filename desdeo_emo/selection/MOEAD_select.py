import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_tools.scalarization.MOEADSF import (
    MOEADSFBase,
    Tchebycheff,
    PBI,
    WeightedSum,
)


class MOEAD_select(SelectionBase):
    """The MOEAD selection operator. 

    Parameters
    ----------
    pop : Population
        The population of individuals
    SF_type : MOEADSFBase
        The scalarizing function employed to evaluate the solutions

    """

    def __init__(self, pop: Population, SF_type: MOEADSFBase):
        # initialize
        self.SF_type = SF_type

    def do(
        self,
        pop: Population,
        vectors: ReferenceVectors,
        ideal_point,
        current_neighborhood,
        offspring_fx,
    ) -> List[int]:
        """Select the individuals that are kept in the neighborhood.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.
        ideal_point
            Ideal vector found so far
        current_neighborhood
            Neighborhood to be updated
        offspring_fx
            Offspring solution to be compared with the rest of the neighborhood

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        # Compute the value of the SF for each neighbor
        num_neighbors = len(current_neighborhood)
        current_population = pop.fitness[current_neighborhood, :]
        current_reference_vectors = vectors.values_planar[current_neighborhood, :]
        offspring_population = np.tile(offspring_fx, (num_neighbors, 1))
        ideal_point_matrix = np.tile(ideal_point, (num_neighbors, 1))

        values_SF = self._evaluate_SF(
            current_population, current_reference_vectors, ideal_point_matrix
        )
        values_SF_offspring = self._evaluate_SF(
            offspring_population, current_reference_vectors, ideal_point_matrix
        )

        # Compare the offspring with the individuals in the neighborhood
        # and replace the ones which are outperformed by it.
        selection = np.where(values_SF_offspring < values_SF)[0]

        return current_neighborhood[selection]

    def _evaluate_SF(self, neighborhood, weights, ideal_point):
        num_neighbors = len(neighborhood)
        num_objectives = np.shape(weights)[1]
        SF_values = []

        # Replace the zeros in the weight vectors with a small value to avoid
        # errors when computing the scalarization function
        fixed_reference_vectors = weights
        fixed_reference_vectors[fixed_reference_vectors == 0] = 0.0001

        # Get the value of the selected scalarization function
        SF_values = np.array(
            list(map(self.SF_type, neighborhood, fixed_reference_vectors, ideal_point))
        )
        return SF_values

