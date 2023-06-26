import numpy as np
import pytest
from desdeo_emo.recombination import BP_mutation

# Define a fixture for BP_mutation
@pytest.fixture
def bp_mutation():
    # Set lower and upper limits for the BP_mutation
    lower_limits = np.array([0, 0, 0])
    upper_limits = np.array([10, 10, 10])
    return BP_mutation(lower_limits, upper_limits)

# Test the initialization of BP_mutation
def test_initialization(bp_mutation):
    # Check if the lower_limits property is correctly set
    assert np.array_equal(bp_mutation.lower_limits, np.array([0, 0, 0]))
    # Check if the upper_limits property is correctly set
    assert np.array_equal(bp_mutation.upper_limits, np.array([10, 10, 10]))
    # Check if the ProM property is approximately equal to 1/3
    assert pytest.approx(bp_mutation.ProM) == 0.3333333333333333
    # Check if the DisM property is equal to 20
    assert bp_mutation.DisM == 20
    print("Initializing works!")

# Test the do() method of BP_mutation
def test_do(bp_mutation):
    # Create an offspring array
    offspring = np.array([5, 5, 5])
    # Perform the mutation using the BP_mutation
    mutated_offspring = bp_mutation.do(offspring)
    # Check if all mutated_offspring values are greater than or equal to the lower limits
    assert np.all(mutated_offspring >= bp_mutation.lower_limits)
    # Check if all mutated_offspring values are less than or equal to the upper limits
    assert np.all(mutated_offspring <= bp_mutation.upper_limits)
