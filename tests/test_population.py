import pytest
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_problem.problem import MOProblem
from desdeo_problem.problem import Variable, ScalarObjective

@pytest.fixture
def problem():
    # Define the objective functions
    def f_1(x):
        x = np.atleast_2d(x)
        return -4.07 - 2.27 * x[:, 0]

    def f_2(x):
        x = np.atleast_2d(x)
        return -2.60 - 0.03 * x[:, 0] - 0.02 * x[:, 1] - 0.01 / (1.39 - x[:, 0]**2) - 0.30 / (1.39 + x[:, 1]**2)

    # Define the objectives and variables
    objective_1 = ScalarObjective(name="f_1", evaluator=f_1)
    objective_2 = ScalarObjective(name="f_2", evaluator=f_2)
    objectives = [objective_1, objective_2]
    variables = [
        Variable(name="x_1", initial_value=0.5, lower_bound=0.3, upper_bound=1.0),
        Variable(name="x_2", initial_value=0.5, lower_bound=0.3, upper_bound=1.0)
    ]

    # Create a mock MOProblem
    problem = MOProblem(variables=variables, objectives=objectives)
    return problem

def test_population_creation(problem):
    # Create a Population instance
    pop_size = 10
    population = Population(problem, pop_size)
    # Assert that the population size is correct
    assert len(population.individuals) == pop_size

def test_population_add(problem):
    # Create a Population instance
    pop_size = 10
    population = Population(problem, pop_size)

    # Create mock offsprings
    offsprings = np.random.rand(pop_size, problem.n_of_variables)

    # Call the add method
    results = population.add(offsprings)

    # Assert that the individuals and objectives are updated correctly
    assert len(population.individuals) == 2 * pop_size
    assert population.objectives.shape == (2 * pop_size, problem.n_of_objectives)

    # Assert that the fitness array has the correct shape
    assert population.fitness.shape == (2 * pop_size, problem.n_of_objectives)

    # Assert that the results contain the expected information
    assert results.objectives.shape == (pop_size, problem.n_of_objectives)
    assert results.fitness.shape == (pop_size, problem.n_of_objectives)

def test_population_keep(problem):
    # Create a Population instance
    pop_size = 10
    population = Population(problem, pop_size)

    # Create a list of indices to keep
    indices_to_keep = [0, 2, 4, 6, 8]

    # Call the keep method
    population.keep(indices_to_keep)

    # Assert that the population size is updated correctly
    assert len(population.individuals) == len(indices_to_keep)
    assert population.objectives.shape[0] == len(indices_to_keep)
    assert population.fitness.shape[0] == len(indices_to_keep)

def test_population_delete(problem):
    # Create a Population instance
    pop_size = 10
    population = Population(problem, pop_size)

    # Create a list of indices to delete
    indices_to_delete = [1, 3, 5, 7, 9]

    # Call the delete method
    population.delete(indices_to_delete)

    # Assert that the population size is updated correctly
    assert len(population.individuals) == pop_size - len(indices_to_delete)
    assert population.objectives.shape[0] == pop_size - len(indices_to_delete)
    assert population.fitness.shape[0] == pop_size - len(indices_to_delete)

def test_population_mate(problem):
    # Create a Population instance
    pop_size = 10
    population = Population(problem, pop_size)

    # Call the mate method
    offspring = population.mate()

    # Assert that the offspring size is correct
    assert len(offspring) == pop_size

    # Assert that the offspring has the correct shape
    assert offspring.shape == population.individuals.shape



