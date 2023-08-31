import numpy as np
import pytest
from random import seed
from desdeo_emo.recombination import SinglePoint_Xover, SinglePoint_Mutation

# Test case for SinglePoint_Xover's crossover method
def test_SinglePoint_Xover_crossover():
    xover = SinglePoint_Xover()
    ind_0 = np.array([0, 1, 0, 1, 0])
    ind_1 = np.array([1, 0, 1, 0, 1])
    new_0, new_1 = xover.crossover(ind_0, ind_1)
    assert len(new_0) == len(ind_0)
    assert len(new_1) == len(ind_1)

# Test case for SinglePoint_Xover's mutation method
def test_SinglePoint_Xover_mutation():
    xover = SinglePoint_Xover()
    indi = np.array([0, 1, 0, 1, 0])
    mutated = xover.mutation(indi)
    assert len(mutated) == len(indi)

# Test case for SinglePoint_Xover's do method
def test_SinglePoint_Xover_do():
    xover = SinglePoint_Xover()
    pop = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    offspring = xover.do(pop)
    assert offspring.shape == pop.shape

# Test case for SinglePoint_Mutation's mutation method
def test_SinglePoint_Mutation_mutation():
    mutation = SinglePoint_Mutation()
    indi = np.array([0, 1, 0, 1, 0])
    mutated = mutation.mutation(indi)
    assert len(mutated) == len(indi)

# Test case for SinglePoint_Mutation's do method
def test_SinglePoint_Mutation_do():
    mutation = SinglePoint_Mutation()
    offspring = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    mutated_offspring = mutation.do(offspring)
    assert mutated_offspring.shape == offspring.shape

# Test case for SinglePoint_Xover's crossover method with single-element arrays
def test_SinglePoint_Xover_crossover_single_element_array():
    xover = SinglePoint_Xover()
    ind_0 = np.array([0])
    ind_1 = np.array([1])
    new_0, new_1 = xover.crossover(ind_0, ind_1)
    assert len(new_0) == 1
    assert len(new_1) == 1

# Test case for SinglePoint_Xover's crossover method with large arrays
def test_SinglePoint_Xover_crossover_large_array():
    xover = SinglePoint_Xover()
    ind_0 = np.ones(10000)
    ind_1 = np.zeros(10000)
    new_0, new_1 = xover.crossover(ind_0, ind_1)
    assert len(new_0) == 10000
    assert len(new_1) == 10000
