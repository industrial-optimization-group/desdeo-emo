import numpy as np
import pytest

from desdeo_tools.scalarization import GuessASF


@pytest.fixture
def simple_discrete_data():
    solutions = np.array(
        [[-1, -1, -1], [0.5, -1.5, 1.5], [-1.6, 2.1, -1.2], [-0.8, -0.9, -2.2]],
        dtype=float,
    )

    ideal = np.min(solutions, axis=0)
    nadir = np.max(solutions, axis=0)

    return (solutions, ideal, nadir)


def test_guess(simple_discrete_data):
    solutions, _, nadir = simple_discrete_data

    asf = GuessASF(nadir)

    # should be first solution
    ref_point = np.array([-1.1, -1.3, 0.5])
    res = asf(solutions, ref_point)
    min_i = np.argmin(res)

    assert min_i == 0

    # should be second solution
    ref_point = np.array([0.4, -1.3, 1.6])
    res = asf(solutions, ref_point)
    min_i = np.argmin(res)

    assert min_i == 1

    # should be third solution
    ref_point = np.array([-1.7, 2.2, -1.3])
    res = asf(solutions, ref_point)
    min_i = np.argmin(res)

    assert min_i == 2

    # should be second fourth
    ref_point = np.array([-0.7, -0.8, -2.0])
    res = asf(solutions, ref_point)
    min_i = np.argmin(res)

    assert min_i == 3
