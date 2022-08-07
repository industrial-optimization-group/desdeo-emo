from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.vehicle_crashworthiness
def test_number_of_variables():
    p: MOProblem = vehicle_crashworthiness()

    assert p.n_of_variables == 5

@pytest.mark.vehicle_crashworthiness
def test_number_of_objectives():
    p: MOProblem = vehicle_crashworthiness()

    assert p.n_of_objectives == 3

# Evaluate the problem with some variable values
@pytest.mark.vehicle_crashworthiness
def test_car_crash():
    p: MOProblem = vehicle_crashworthiness()

    xs = np.array([[2, 2, 2, 2, 2],[1, 2, 2, 2, 3]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[1683.133345, 9.6266, 0.1233],[1685.231967, 9.4613, 0.1038]])

    npt.assert_allclose(objective_vectors, expected_result)

# Testing error if variable values are not in between of lower and upper bounds.
@pytest.mark.vehicle_crashworthiness
def test_variable_bounds_error():
    with pytest.raises(ValueError):
        p: MOProblem = vehicle_crashworthiness(var_iv=np.array([2, 3, 0.9, 2, 4]))
