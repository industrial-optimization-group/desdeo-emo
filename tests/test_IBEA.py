import pytest
import numpy as np
#linter should remove this: 
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.population.Population import Population

from desdeo_emo.EAs.IBEA import IBEA
from desdeo_problem.problem import Variable, ScalarObjective
from desdeo_problem import variable_builder, ScalarObjective, MOProblem

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

# Test the initialization of IBEA
def test_IBEA_initialization(problem):
    # Create an instance of IBEA
    ibea = IBEA(problem, population_size=32, n_iterations=10, n_gen_per_iter=100, total_function_evaluations=10000)

    # Perform assertions to verify the initialization
    assert ibea.n_iterations == 10
    assert ibea.n_gen_per_iter == 100
    assert ibea.total_function_evaluations == 10000
    # Add more assertions if needed

    # Additional checks for default values
    assert ibea.kappa == 0.05

def test_IBEA(problem):
    # Create an instance of IBEA
    #ib = IBEA(problem, population_size=100, n_iterations=10, n_gen_per_iter=100,total_function_evaluations=5000)
    ib = IBEA(problem, n_iterations=10, n_gen_per_iter=100, population_size=100, total_function_evaluations=5000)

    while ib.continue_evolution():
        ib.iterate()
    individuals, solutions = ib.end()
    #print("individuals ", individuals.shape[0])
    #print("objective_values: ", solutions[0])

    assert individuals.shape[0] > 0
    assert len(solutions) > 0

   
