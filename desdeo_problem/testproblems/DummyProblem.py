from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem

import numpy as np

def dummy_problem() -> MOProblem:
    """An example on how to implement a problem with 3 objectives and 4 variables and no constraints.

    Returns:
        MOProblem: a problem object.
    """
    def f1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        # sum the rows
        return np.sum(x, axis=1)

    def f2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        return -np.sum(x, axis=1)

    def f3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        # take the product of the rows
        return np.prod(x, axis=1)

    # args: name of objective, evaluator
    objective_1 = ScalarObjective("F1", f1)
    objective_2 = ScalarObjective("F2", f2)
    objective_3 = ScalarObjective("F3", f3)

    objectives = [objective_1, objective_2, objective_3]

    # args: name of variable, initial value, lower bound, upper bound
    variable_1 = Variable("x1", 3, 0, 5)
    variable_2 = Variable("x2", 3, 2, 4)
    variable_3 = Variable("x3", -2, -3, 4)
    variable_4 = Variable("x4", -1, -6, 4)

    variables = [variable_1, variable_2, variable_3, variable_4]

    # instantiate MOProblem object
    problem = MOProblem(objectives, variables)

    return problem

    