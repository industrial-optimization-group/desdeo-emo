import pytest
import numpy as np
from desdeo_emo.population.CreateIndividuals import create_new_individuals
from desdeo_problem.problem import MOProblem, Variable, ScalarObjective

# Define the objective functions
def f_1(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return -4.07 - 2.27 * x[:, 0]

def f_2(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return -2.60 - 0.03 * x[:, 0] - 0.02 * x[:, 1] - 0.01 / (1.39 - x[:, 0]**2) - 0.30 / (1.39 + x[:, 1]**2)


def test_create_individuals():
    objective_1 = ScalarObjective(name="f_1", evaluator=f_1)
    objective_2 = ScalarObjective(name="f_2", evaluator=f_2)
    objectives = [objective_1, objective_2]

    x1 = Variable(name="x_1", initial_value=0.5, lower_bound=0.3, upper_bound=1.0)
    x2 = Variable(name="x_2", initial_value=0.5, lower_bound=0.3, upper_bound=1.0)
    x3 = Variable(name="x_3", initial_value=0.5, lower_bound=0.3, upper_bound=1.0)
    
    variables1 = [x1]
    variables2 = [x1, x2]
    variables3 = [x1, x2, x3]

    problem = MOProblem(variables=variables3, objectives=objectives)

    # Test case 1: RandomDesign
    individuals = create_new_individuals("RandomDesign", problem, pop_size=10)
    assert individuals.shape == (10, 3)

    # Test case 2: LHSDesign
    individuals = create_new_individuals("LHSDesign", problem, pop_size=10)
    assert individuals.shape == (10, 3)
