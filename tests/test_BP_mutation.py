import numpy as np
import pytest
from desdeo_emo.recombination import BP_mutation


@pytest.fixture
def bp_mutation():
    lower_limits = np.array([0, 0, 0])
    upper_limits = np.array([10, 10, 10])
    return BP_mutation(lower_limits, upper_limits)


def test_initialization(bp_mutation):
    assert np.array_equal(bp_mutation.lower_limits, np.array([0, 0, 0]))
    assert np.array_equal(bp_mutation.upper_limits, np.array([10, 10, 10]))
    assert pytest.approx(bp_mutation.ProM) == 0.3333333333333333
    assert bp_mutation.DisM == 20


def test_do(bp_mutation):
    offspring = np.array([5, 5, 5])
    mutated_offspring = bp_mutation.do(offspring)
    assert np.all(mutated_offspring >= bp_mutation.lower_limits)
    assert np.all(mutated_offspring <= bp_mutation.upper_limits)



