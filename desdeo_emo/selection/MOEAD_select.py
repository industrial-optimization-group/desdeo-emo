import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import InteractiveDecompositionSelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_tools.scalarization.MOEADSF import (
    MOEADSFBase,
)

from scipy.spatial import distance_matrix


class MOEAD_select(InteractiveDecompositionSelectionBase):
    """The MOEAD selection operator.

    Parameters
    ----------
    pop : Population
        The population of individuals
    SF_type : MOEADSFBase
        The scalarizing function employed to evaluate the solutions

    """

    def __init__(
        self,
        pop: Population,
        SF_type: MOEADSFBase,
        n_neighbors: int,
        selection_type: str = None,
    ):
        # initialize
        super().__init__(pop.pop_size, pop.problem.n_of_fitnesses, selection_type)

        self.n_neighbors = n_neighbors
        # Compute the distance between each pair of reference vectors
        distance_matrix_vectors = distance_matrix(
            self.vectors.values_planar, self.vectors.values_planar
        )
        # Get the closest vectors to obtain the neighborhoods
        self.neighborhoods = np.argsort(
            distance_matrix_vectors, axis=1, kind="quicksort"
        )[:, :n_neighbors]
        self.SF_type = SF_type

    def do(
        self,
        pop: Population,
        current_neighborhood: int,
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
        # num_neighbors = len(current_neighborhood) # TODO: check if self.n_neighbors can be used instead.
        current_population = pop.fitness[self.neighborhoods[current_neighborhood], :]
        offspring_fitness = pop.fitness[-1]
        current_reference_vectors = self.vectors.values_planar[
            self.neighborhoods[current_neighborhood], :
        ]
        offspring_population = np.tile(offspring_fitness, (self.n_neighbors, 1))
        ideal_point_matrix = np.tile(pop.ideal_fitness_val, (self.n_neighbors, 1))

        values_SF = self._evaluate_SF(
            current_population, current_reference_vectors, ideal_point_matrix
        )
        values_SF_offspring = self._evaluate_SF(
            offspring_population, current_reference_vectors, ideal_point_matrix
        )

        # Compare the offspring with the individuals in the neighborhood
        # and replace the ones which are outperformed by it.
        selection = np.where(values_SF_offspring < values_SF)[0]

        return self.neighborhoods[current_neighborhood][selection]

    def _evaluate_SF(self, neighborhood, weights, ideal_point):
        # Replace the zeros in the weight vectors with a small value to avoid
        # errors when computing the scalarization function
        fixed_reference_vectors = weights
        fixed_reference_vectors[
            fixed_reference_vectors == 0
        ] = 0.0001  # TODO replace with np.eps?

        # Get the value of the selected scalarization function
        SF_values = np.array(
            list(map(self.SF_type, neighborhood, fixed_reference_vectors, ideal_point))
        )
        return SF_values

    def choose_parents(self, current_neighborhood: int, n_parents: int) -> List[int]:
        current_neighborhood_members = self.neighborhoods[current_neighborhood, :]
        selected_parents = np.random.choice(
            current_neighborhood_members, n_parents, replace=False
        )
        return selected_parents

    def adapt_RVs(self, fitness: np.ndarray) -> None:
        super().adapt_RVs(fitness)
        # Recompute the distance between each pair of reference vectors
        distance_matrix_vectors = distance_matrix(
            self.vectors.values_planar, self.vectors.values_planar
        )
        # Get the closest vectors to obtain the neighborhoods
        self.neighborhoods = np.argsort(
            distance_matrix_vectors, axis=1, kind="quicksort"
        )[:, : self.n_neighbors]
