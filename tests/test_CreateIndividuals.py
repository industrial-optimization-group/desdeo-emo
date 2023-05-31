import pytest
import numpy as np
from desdeo_emo.population.CreateIndividuals import create_new_individuals
from desdeo_problem.problem import MOProblem, Variable, ScalarObjective

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

    problem = MOProblem(variables=variables, objectives=objectives)
    # Create a mock MOProblem
    return problem

def test_create_individuals(problem):
    # Test case 1: RandomDesign
    individuals = create_new_individuals("RandomDesign", problem, pop_size=10)
    assert individuals.shape == (10, 2)

    # Test case 2: LHSDesign
    individuals = create_new_individuals("LHSDesign", problem, pop_size=10)
    assert individuals.shape == (10, 2)

    #Need to add more tests here.