from pyRVEA.allclasses import ReferenceVectors
import numpy as np
from itertools import product


class newRV(ReferenceVectors):
    """pass"""

    def rotate_to_axis(self, ref_point):
        self.values = rotate(ref_point, [0, 0, 1], self.values)

    def revert_rotation(self, ref_point):
        self.values = rotate([0, 0, 1], ref_point, self.values)

    def project_to_hyperplane(self):
        """Projects the reference vectors to the hyperplane xn = 1."""
        self.values[:, -1] = 1

    def translate_to_hypersphere(self):
        """Reverse of preject_to_hyperplane()."""
        self.values[:, -1] = np.sqrt(
            1 - np.sum(np.square(self.values[:, 0:-1]), axis=1)
        )

    def interact_v2(self, ref_point):
        """New kind of interaction."""
        self.rotate_to_axis(ref_point)
        self.project_to_hyperplane()
        newvals = dist_based_translation(self.values[:, 0:-1])
        self.values[:, 0:-1] = newvals
        self.translate_to_hypersphere()
        self.revert_rotation(ref_point)

    def interact_v3(self, ref_point):
        """New kind of interaction. More coverage."""
        self.values = np.zeros(self.values.shape)
        newvals = []
        combinations = np.asarray(list(product([-1, 1], repeat=self.values.shape[1])))
        combinations[:, -1] = np.unique(abs(combinations[:, -1]), axis=0)
        for combination in combinations:
            changedvals = np.copy(self.initial_values) * combination
            if newvals == []:
                newvals = changedvals
            else:
                newvals = np.vstack((newvals, changedvals))
        newvals = dist_based_translation(newvals[:, 0:-1])
        changedvals = np.ones((newvals.shape[0], newvals.shape[1] + 1))
        changedvals[:, 0:-1] = newvals
        self.values = changedvals
        self.translate_to_hypersphere()
        self.revert_rotation(ref_point)
        num_rows = self.values.shape[0]
        delete_rows = []
        for index in range(num_rows):
            if np.any(self.values[index, :] < 0):
                if delete_rows == []:
                    delete_rows = np.asarray(index)
                else:
                    delete_rows = np.hstack((delete_rows, index))
        self.values = np.delete(self.values, delete_rows, axis=0)


def rotate(initial_vector, rotated_vector, other_vectors):
    """Calculate the rotation matrix that rotates the initial_vector to the
    rotated_vector. Apply that rotation on other_vectors and return.
    Uses Householder reflections twice to achieve this."""

    init_vec_norm = normalize(initial_vector)
    rot_vec_norm = normalize(rotated_vector)
    middle_vec_norm = normalize(init_vec_norm + rot_vec_norm)
    first_reflector = init_vec_norm - middle_vec_norm
    second_reflector = middle_vec_norm - rot_vec_norm
    Q1 = householder(first_reflector)
    Q2 = householder(second_reflector)
    reflection_matrix = np.matmul(Q2, Q1)
    rotated_vectors = np.matmul(other_vectors, np.transpose(reflection_matrix))
    return rotated_vectors


def normalize(vector):
    """Normalize and return a vector."""
    return vector / np.linalg.norm(vector)


def householder(vector):
    """Return reflection matrix via householder transformation."""
    identity_mat = np.eye(len(vector))
    v = vector[np.newaxis]
    denominator = np.matmul(v, v.T)
    numerator = np.matmul(v.T, v)
    rot_mat = identity_mat - (2 * numerator / denominator)
    return rot_mat


def dist_based_translation(vectors):
    """Translates points towards origin based on distance."""
    dist = np.sqrt(np.sum(np.square(vectors), axis=1))
    max_dist = np.amax(dist)
    # max_dist = 1
    alpha = 200
    ratio = alpha * (1 / (dist * dist) - 1 / (max_dist * max_dist))
    t_factor = 1 / (1 + ratio)
    return vectors * t_factor[np.newaxis].T


rv = newRV(15, 3)
rv.interact_v3([1, 1, 1])
rv.plot_ref_V()
