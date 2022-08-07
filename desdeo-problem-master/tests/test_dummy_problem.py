from desdeo_problem.testproblems import dummy_problem
from desdeo_problem.problem import ProblemError
import numpy as np
import numpy.testing as npt

def test_dummy_problem():
    # create a dummy problem
    problem = dummy_problem()

    # evaluate the problem with some variable values
    xs = np.array([[1, 3, -2, -5], [1.1, 3.3, -2.2, -5.5]])

    objective_vectors = problem.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[-3, 3, 30], [-3.3, 3.3, 43.923]])

    npt.assert_allclose(objective_vectors, expected_result)
