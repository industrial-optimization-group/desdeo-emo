from itertools import combinations, product

import numpy as np
from scipy.special import comb


def normalize(vectors):
    """
    Normalize a set of vectors.

    The length of the returned vectors will be unity.

    Parameters
    ----------
    vectors : np.ndarray
        Set of vectors of any length, except zero.

    """
    if len(np.asarray(vectors).shape) == 1:
        return vectors / np.linalg.norm(vectors)
    norm = np.linalg.norm(vectors, axis=1)
    return vectors / norm[:, np.newaxis]


def shear(vectors, degrees: float = 5):
    """
    Shear a set of vectors lying on the plane z=0 towards the z-axis, such that the
    resulting vectors 'degrees' angle away from the z axis.

    z is the last element of the vector, and has to be equal to zero.

    Parameters
    ----------
    vectors : numpy.ndarray
        The final element of each vector should be zero.
    degrees : float, optional
        The angle that the resultant vectors make with the z axis. Unit is radians.
        (the default is 5)

    """
    angle = degrees * np.pi / 180
    m = 1 / np.tan(angle)
    norm = np.linalg.norm(vectors, axis=1)
    vectors[:, -1] += norm * m
    return normalize(vectors)


def rotate(initial_vector, rotated_vector, other_vectors):
    """Calculate the rotation matrix that rotates the initial_vector to the
    rotated_vector. Apply that rotation on other_vectors and return.
    Uses Householder reflections twice to achieve this."""

    init_vec_norm = normalize(initial_vector)
    rot_vec_norm = normalize(np.asarray(rotated_vector))
    middle_vec_norm = normalize(init_vec_norm + rot_vec_norm)
    first_reflector = init_vec_norm - middle_vec_norm
    second_reflector = middle_vec_norm - rot_vec_norm
    Q1 = householder(first_reflector)
    Q2 = householder(second_reflector)
    reflection_matrix = np.matmul(Q2, Q1)
    rotated_vectors = np.matmul(other_vectors, np.transpose(reflection_matrix))
    return rotated_vectors


def householder(vector):
    """Return reflection matrix via householder transformation."""
    identity_mat = np.eye(len(vector))
    v = vector[np.newaxis]
    denominator = np.matmul(v, v.T)
    numerator = np.matmul(v.T, v)
    rot_mat = identity_mat - (2 * numerator / denominator)
    return rot_mat


def rotate_toward(initial_vector, final_vector, other_vectors, degrees: float = 5):
    """
    Rotate other_vectors (with the centre at initial_vector) towards final_vector
    by an angle degrees.

    Parameters
    ----------
    initial_vector : np.ndarray
        Centre of the vectors to be rotated.
    final_vector : np.ndarray
        The final position of the center of other_vectors.
    other_vectors : np.ndarray
        The array of vectors to be rotated
    degrees : float, optional
        The amount of rotation (the default is 5)

    Returns
    -------
    rotated_vectors : np.ndarray
        The rotated vectors
    reached: bool
        True if final_vector has been reached
    """
    final_vector = normalize(final_vector)
    initial_vector = normalize(initial_vector)
    cos_phi = np.dot(initial_vector, final_vector)
    theta = degrees * np.pi / 180
    cos_theta = np.cos(theta)
    phi = np.arccos(cos_phi)
    if phi < theta:
        return (rotate(initial_vector, final_vector, other_vectors), True)
    cos_phi_theta = np.cos(phi - theta)
    A = np.asarray([[cos_phi, 1], [1, cos_phi]])
    B = np.asarray([cos_phi_theta, cos_theta])
    x = np.linalg.solve(A, B)
    rotated_vector = x[0] * initial_vector + x[1] * final_vector
    return (rotate(initial_vector, rotated_vector, other_vectors), False)


class ReferenceVectors:
    """Class object for reference vectors."""

    def __init__(
        self,
        lattice_resolution: int = None,
        number_of_objectives: int = None,
        creation_type: str = "Uniform",
        vector_type: str = "Spherical",
        ref_point: list = None,
    ):
        """Create a Reference vectors object.

        A simplex lattice is formed

        Parameters
        ----------
        lattice_resolution : int
            Number of divisions along an axis when creating the simplex lattice.
        number_of_objectives : int
            Number of objectives.
        creation_type : str, optional
            'Uniform' creates the reference vectors uniformly using simplex lattice
            design. 'Focused' creates reference vectors symmetrically around a central
            reference vector. 'Reversed' coming soon.By default 'Uniform'.
        vector_type : str, optional
            'Spherical' normalizes the vectors to a hypersphere, i.e. the second norm
            is equal to 1. 'Planar' normalizes vectors to a plane, i.e. the first norm
            is equal to 1. By default 'Spherical'.
        ref_point : list, optional
            User preference information for a priori methods.
        """

        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_vectors = 0
        self.creation_type = creation_type
        self.vector_type = vector_type
        self.values = []
        self.values_planar = []
        self.ref_point = [1] * number_of_objectives if ref_point is None else ref_point
        self._create(creation_type)
        self.initial_values = np.copy(self.values)
        self.initial_values_planar = np.copy(self.values_planar)
        self.neighbouring_angles()
        # self.iteractive_adapt_1() Can use this for a priori preferences!

    def _create(self, creation_type: str = "Uniform"):
        """Create the reference vectors.

        Parameters
        ----------
        creation_type : str, optional
            'Uniform' creates the reference vectors uniformly using simplex lattice
            design. 'Focused' creates reference vectors symmetrically around a central
            reference vector. By default 'Uniform'.
        """
        if creation_type == "Uniform":
            number_of_vectors = comb(
                self.lattice_resolution + self.number_of_objectives - 1,
                self.number_of_objectives - 1,
                exact=True,
            )
            self.number_of_vectors = number_of_vectors
            temp1 = range(1, self.number_of_objectives + self.lattice_resolution)
            temp1 = np.array(list(combinations(temp1, self.number_of_objectives - 1)))
            temp2 = np.array(
                [range(self.number_of_objectives - 1)] * self.number_of_vectors
            )
            temp = temp1 - temp2 - 1
            weight = np.zeros(
                (self.number_of_vectors, self.number_of_objectives), dtype=int
            )
            weight[:, 0] = temp[:, 0]
            for i in range(1, self.number_of_objectives - 1):
                weight[:, i] = temp[:, i] - temp[:, i - 1]
            weight[:, -1] = self.lattice_resolution - temp[:, -1]
            self.values = weight / self.lattice_resolution
            self.values_planar = np.copy(self.values)
            self.normalize()
            return
        elif creation_type == "Focused":
            point_set = [[0, 1, -1]] * (self.number_of_objectives - 1)
            # The cartesian product of point_set.
            initial = np.array(list(product(*point_set)))[1:]
            # First element was removed because of the error during normalization.
            initial = normalize(initial)
            initial = np.hstack((initial, np.zeros((initial.shape[0], 1))))
            final = shear(initial, degrees=5)
            # Adding the first element back
            final = np.vstack(([0] * (self.number_of_objectives - 1) + [1], final))
            self.number_of_vectors = final.shape[0]
            self.values = rotate(final[0], self.ref_point, final)
            self.values_planar = np.copy(self.values)
            self.normalize()
            self.add_edge_vectors()
        elif creation_type == "Sparse_Focused":
            initial = np.eye(self.number_of_objectives - 1)
            initial = np.vstack((initial, -initial))
            initial = normalize(initial)
            initial = np.hstack((initial, np.zeros((initial.shape[0], 1))))
            final = shear(initial, degrees=5)
            # Adding the first element back
            final = np.vstack(([0] * (self.number_of_objectives - 1) + [1], final))
            self.number_of_vectors = final.shape[0]
            self.values = rotate(final[0], self.ref_point, final)
            self.values_planar = np.copy(self.values)
            self.normalize()
            self.add_edge_vectors()

    def normalize(self):
        """Normalize the reference vectors to a unit hypersphere."""
        self.number_of_vectors = self.values.shape[0]
        norm_2 = np.linalg.norm(self.values, axis=1)
        norm_1 = np.sum(self.values_planar, axis=1)
        norm_2 = np.repeat(norm_2, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives
        )
        norm_1 = np.repeat(norm_1, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives
        )
        norm_2[norm_2 == 0] = np.finfo(float).eps
        self.values = np.divide(self.values, norm_2)
        self.values_planar = np.divide(self.values_planar, norm_1)

    def neighbouring_angles(self) -> np.ndarray:
        """Calculate neighbouring angles for normalization."""
        cosvv = np.dot(self.values, self.values.transpose())
        cosvv.sort(axis=1)
        cosvv = np.flip(cosvv, 1)
        cosvv[cosvv > 1] = 1
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
        self.values_planar = self.initial_values_planar * translation_param + (
            (1 - translation_param) * ref_point
        )
        self.normalize()

    def slow_interactive_adapt(self, ref_point):
        """Basically a wrapper around rotate_toward. Slowly rotate ref vectors toward
        ref_point. Return a boolean value to tell if the ref_point has been reached.

        Parameters
        ----------
        ref_point : list or np.ndarray
            The reference vectors will slowly move towards the ref_point.

        Returns
        -------
        boolean
            True if ref_point has been reached. False otherwise.
        """
        assert self.creation_type == "Focused" or self.creation_type == "Sparse_Focused"
        if np.array_equal(self.values[0], ref_point):
            return
        self.values, reached = rotate_toward(
            self.values[0], ref_point, self.values[0 : -self.number_of_objectives]
        )
        self.values_planar = self.values
        self.add_edge_vectors()
        self.normalize()
        return reached

    def add_edge_vectors(self):
        """Add edge vectors to the list of reference vectors.

        Used to cover the entire orthant when preference information is
        provided.

        """
        edge_vectors = np.eye(self.values.shape[1])
        self.values = np.vstack([self.values, edge_vectors])
        self.values_planar = np.vstack([self.values_planar, edge_vectors])
        self.number_of_vectors = self.values.shape[0]
        self.normalize()
