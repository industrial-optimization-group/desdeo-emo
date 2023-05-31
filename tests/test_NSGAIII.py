import pytest
import numpy as np
from desdeo_emo.EAs.NSGAIII import NSGAIII
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



# Test the initialization of NSGAIII
def test_nsga3_initialization(problem):
    # Test case 1: Check default parameter values
    nsga = NSGAIII(
        problem=problem,
        n_iterations=10,
        n_gen_per_iter=100,
        population_size=100
    )

    assert all(
        obj.name == prob_obj.name for obj, prob_obj in zip(nsga.population.problem.objectives, problem.objectives)
    )

    assert len(nsga.population.individuals) == 100
    assert nsga.selection_type is None
    assert nsga.interact is False
    assert nsga.use_surrogates is False
    assert nsga.n_iterations == 10
    assert nsga.n_gen_per_iter == 100
    assert nsga.total_function_evaluations == 0
    assert nsga.keep_archive is False
    assert nsga.save_non_dominated is False



