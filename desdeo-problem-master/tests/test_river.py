from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

# Testing number of variables
@pytest.mark.river_pollution
def test_number_of_variables():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_variables == 2

# Testing problem with four objectives.
@pytest.mark.river_pollution
def test_four_objective_problem():
    p: MOProblem = river_pollution_problem(five_obj=False)

    assert p.n_of_objectives == 4

# Testing default problem is with five objectives.
@pytest.mark.river_pollution
def test_five_objective_problem():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_objectives == 5

# Evaluating problem with some variable values.
@pytest.mark.river_pollution
def test_problem():
    p: MOProblem = river_pollution_problem()

    # Variable values
    xs = np.array([[1, 0.5],[0.7, 0.8]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[-6.34, -2.92879892, -0.321111, 0.182857, 0.35],[-5.659, -3.04811111, -7.02666666, 1.1733333, 0.15]])

    npt.assert_allclose(objective_vectors, expected_result, rtol=1e-6)

# Testing error if variable values are not in between of lower and upper bounds.
@pytest.mark.river_pollution
def test_variable_bounds_error():
    with pytest.raises(ValueError):
        p: MOProblem = river_pollution_problem(var_iv=np.array([0, 0]))