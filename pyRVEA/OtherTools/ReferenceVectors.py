from itertools import combinations

import numpy as np
from scipy.special import comb


class ReferenceVectors:
    """Class object for reference vectors."""

    def __init__(self, lattice_resolution: int, number_of_objectives: int):
        """Create a Reference vectors object.

        A simplex lattice is formed

        Parameters
        ----------
        lattice_resolution : int
            Number of divisions along an axis when creating the simplex lattice.
        number_of_objectives : int
            Number of objectives.
        """
        number_of_vectors = comb(
            lattice_resolution + number_of_objectives - 1,
            number_of_objectives - 1,
            exact=True,
        )
        temp1 = range(1, number_of_objectives + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, number_of_objectives - 1)))
        temp2 = np.array([range(number_of_objectives - 1)] * number_of_vectors)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_vectors, number_of_objectives), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, number_of_objectives - 1):
            weight[:, i] = temp[:, i] - temp[:, i - 1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        self.values = weight / lattice_resolution
        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_vectors = number_of_vectors
        self.values_planar = np.copy(self.values)
        self.normalize()
        self.initial_values = np.copy(self.values)
        self.neighbouring_angles()
        # self.iteractive_adapt_1() Can use this for a priori preferences!

    def normalize(self):
        """Normalize the reference vectors to a unit hypersphere."""
        self.number_of_vectors = self.values.shape[0]
        norm = np.linalg.norm(self.values, axis=1)
        norm = np.repeat(norm, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives
        )
        self.values = np.divide(self.values, norm)

    def neighbouring_angles(self) -> np.ndarray:
        """Calculate neighbouring angles for normalization."""
        cosvv = np.dot(self.values, self.values.transpose())
        cosvv.sort(axis=1)
        cosvv = np.flip(cosvv, 1)
        acosvv = np.arccos(cosvv[:, 1])
        self.neighbouring_angles_current = acosvv
        return acosvv

    def adapt(self, fitness: np.ndarray):
        """Adapt reference vectors. Then normalize.

        Parameters
        ----------
        fitness : np.ndarray
        """
        max_val = np.amax(fitness, axis=0)
        min_val = np.amin(fitness, axis=0)
        self.values = np.multiply(
            self.initial_values,
            np.tile(np.subtract(max_val, min_val), (self.number_of_vectors, 1)),
        )
        self.normalize()

    def iteractive_adapt_1(self, ref_point, translation_param=0.2):
        """Adapt reference vectors linearly towards a reference point. Then normalize.

        The details can be found in the following paper: Hakanen, Jussi &
        Chugh, Tinkle & Sindhya, Karthik & Jin, Yaochu & Miettinen, Kaisa.
        (2016). Connections of Reference Vectors and Different Types of
        Preference Information in Interactive Multiobjective Evolutionary
        Algorithms.

        Parameters
        ----------
        ref_point :

        translation_param :
            (Default value = 0.2)

        """
        self.values = self.initial_values * translation_param + (
            (1 - translation_param) * ref_point
        )
        self.normalize()

    def add_edge_vectors(self):
        """Add edge vectors to the list of reference vectors.
     
        Used to cover the entire orthant when preference information is
        provided.

        """
        edge_vectors = np.eye(self.values.shape[1])
        self.values = np.vstack([self.values, edge_vectors])
        self.number_of_vectors = self.values.shape[0]
        self.normalize()

