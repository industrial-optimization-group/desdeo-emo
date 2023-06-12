import numpy as np
from random import shuffle
import pytest
from desdeo_emo.recombination import SBX_xover

# Define a fixture for SBX_xover
@pytest.fixture
def sbx_xover_instance():
    return SBX_xover()

# Test the initialization of SBX_xover
def test_sbx_xover_init():
    # Create an instance of SBX_xover
    sbx_xover = SBX_xover()
    # Check if the ProC property is set to 1
    assert sbx_xover.ProC == 1
    # Check if the DisC property is set to 30
    assert sbx_xover.DisC == 30

# Test the do() method of SBX_xover without mating_pop_ids
def test_sbx_xover_do(sbx_xover_instance):
    # Create a population array
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # Perform the SBX crossover on the population
    offspring = sbx_xover_instance.do(pop)
    # Check if the shape of offspring is the same as the original population
    assert offspring.shape == pop.shape

# Test the do() method of SBX_xover with mating_pop_ids
def test_sbx_xover_do_with_mating_ids(sbx_xover_instance):
    # Create a population array
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # Create a list of mating population IDs
    mating_ids = [2, 1, 3, 0]
    # Perform the SBX crossover on the population with mating IDs
    offspring = sbx_xover_instance.do(pop, mating_pop_ids=mating_ids)
    # Check if the shape of offspring is the same as the original population
    assert offspring.shape == pop.shape

# Test the do() method of SBX_xover with odd number of mating_pop_ids
def test_sbx_xover_do_odd_mating_ids(sbx_xover_instance):
    # Create a population array with odd number of individuals
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Perform the SBX crossover on the population
    offspring = sbx_xover_instance.do(pop)
    # Check if the shape of offspring is (4, 3) due to padding with zeros
    assert offspring.shape == (4, 3)
