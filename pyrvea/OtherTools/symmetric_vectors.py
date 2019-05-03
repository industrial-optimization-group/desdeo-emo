from plotlyanimate import animate_init_, animate_next_
import numpy as np
from itertools import product


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


def normalize(vectors):
    """
    Normalize a set of vectors.

    The length of the returned vectors will be unity.

    Parameters
    ----------
    vectors : np.ndarray
        Set of vectors of any length, except zero.

    """
    if len(vectors.shape) == 1:
        return vectors / np.linalg.norm(vectors)
    norm = np.linalg.norm(vectors, axis=1)
    return vectors / norm[:, np.newaxis]


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


def main():
    initial = np.array(list(product([0, 1, -1], [0, 1, -1])))[1:]
    initial = normalize(initial)
    initial = np.hstack((initial, np.zeros((initial.shape[0], 1))))
    filename = "smooth_transition.html"
    figure = animate_init_(initial, filename)
    final = shear(initial)
    generation = 1
    figure = animate_next_(final, figure, filename, generation)
    complete = False
    while not complete:
        center = np.average(final, axis=0)
        final, complete = rotate_toward(center, np.array([1, 1, 1]), final)
        figure = animate_next_(final, figure, filename, generation)
        generation = generation + 1
    complete = False
    while not complete:
        center = np.average(final, axis=0)
        final, complete = rotate_toward(center, np.array([1, 0, 1]), final)
        figure = animate_next_(final, figure, filename, generation)
        generation = generation + 1
    complete = False
    while not complete:
        center = np.average(final, axis=0)
        final, complete = rotate_toward(center, np.array([0, 0, 1]), final)
        figure = animate_next_(final, figure, filename, generation)
        generation = generation + 1
    complete = False
    while not complete:
        center = np.average(final, axis=0)
        final, complete = rotate_toward(center, np.array([1, 0, 0]), final)
        figure = animate_next_(final, figure, filename, generation)
        generation = generation + 1


if __name__ == "__main__":
    main()
