import numpy as np
from numpy.testing import assert_allclose
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors, normalize, shear, rotate, householder, rotate_toward

def test_init():
    # Test case for creating ReferenceVectors object
    ref_vectors = ReferenceVectors(
        lattice_resolution=4,
        number_of_objectives=3,
        creation_type="Uniform",
        vector_type="Spherical",
        ref_point=[1, 1, 1],
    )

    assert ref_vectors.number_of_objectives == 3
    assert ref_vectors.lattice_resolution == 4
    assert ref_vectors.number_of_vectors == 15
    assert ref_vectors.creation_type == "Uniform"
    assert ref_vectors.vector_type == "Spherical"
    assert ref_vectors.values.shape == (15, 3)
    assert ref_vectors.values_planar.shape == (15, 3)
    assert np.all(ref_vectors.ref_point == [1, 1, 1])
    assert np.all(ref_vectors.initial_values == ref_vectors.values)
    assert np.all(ref_vectors.initial_values_planar == ref_vectors.values_planar)

    print("Initialization works!")


def test_normalize():
    # Test case for normalize function
    vectors = np.array([[1, 2, 3], [4, 5, 6]])
    expected_result = np.array([[0.26726124, 0.53452248, 0.80178373],
                                [0.45584231, 0.56980288, 0.68376346]])
    result = normalize(vectors)
    assert_allclose(result, expected_result)

def test_shear():
    # Test case for shear function
    vectors = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.float64)
    degrees = 45
    expected_result = np.array([[0.31622777, 0.63245553, 0.70710678],
                                [0.42426407, 0.56568542, 0.70710678]])
    result = shear(vectors, degrees)
    assert_allclose(result, expected_result)

def test_rotate():
    # Test case for rotate function
    initial_vector = np.array([1, 0, 0])
    rotated_vector = np.array([0, 1, 0])
    other_vectors = np.array([[1, 1, 0], [1, -1, 0]], dtype=np.float64)
    expected_result = np.array([[-1.,  1.,  0.], [ 1.,  1.,  0.]])
    result = rotate(initial_vector, rotated_vector, other_vectors)
    assert_allclose(result, expected_result)

def test_householder():
    # Test case for householder function
    vector = np.array([1, 1, 1], dtype=np.float64)
    expected_result = np.array([[0.33333333, -0.66666667, -0.66666667], 
                                [-0.66666667,  0.33333333, -0.66666667], 
                                [-0.66666667, -0.66666667,  0.33333333]])
    result = householder(vector)
    assert_allclose(result, expected_result)

def test_rotate_toward():
    # Test case for rotate_toward function
    initial_vector = np.array([1, 0, 0])
    final_vector = np.array([0, 1, 0])
    other_vectors = np.array([[1, 1, 0], [1, -1, 0]], dtype=np.float64)
    degrees = 45
    expected_result = np.array([[7.77156117e-16, 1.41421356e+00, 0.00000000e+00],
                                [1.41421356e+00, -6.66133815e-16, 0.00000000e+00]])
    result, reached = rotate_toward(initial_vector, final_vector, other_vectors, degrees)
    assert_allclose(result, expected_result)
