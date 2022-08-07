from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase

import numpy as np

def vehicle_crashworthiness(var_iv: np.array = np.array([2, 2, 2, 2, 2])) -> MOProblem:
    """The crash safety design problem with 3 objectives.

    Liao, X., Li, Q., Yang, X., Zhang, W. & Li, W. (2007).
    Multiobjective optimization for crash safety design of vehicles
    using stepwise regression model. Structural and multidisciplinary
    optimization, 35(6), 561-569. https://doi.org/10.1007/s00158-007-0163-x

    Arguments:
        var_iv (np.array): Optional, initial variable values. Must be between
            1 and 3. Defaults are [2, 2, 2, 2, 2].

    Returns:
        MOProblem: a problem object.
    """

    if np.any(3 < var_iv) or np.any(var_iv < 1):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Mass
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1640.2823
            + 2.3573285 * x[:, 0]
            + 2.3220035 * x[:, 1]
            + 4.5688768 * x[:, 2]
            + 7.7213633 * x[:, 3]
            + 4.4559504 * x[:, 4]
        )

    # Ain
    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            6.5856
            + 1.15 * x[:, 0]
            - 1.0427 * x[:, 1]
            + 0.9738 * x[:, 2]
            + 0.8364 * x[:, 3]
            - 0.3695 * x[:, 0] * x[:, 3]
            + 0.0861 * x[:, 0] * x[:, 4]
            + 0.3628 * x[:, 1] * x[:, 3]
            - 0.1106 * x[:, 0] ** 2
            - 0.3437 * x[:, 2] ** 2
            + 0.1764 * x[:, 3] ** 2
        )

    # Intrusion
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            -0.0551
            + 0.0181 * x[:, 0]
            + 0.1024 * x[:, 1]
            + 0.0421 * x[:, 2]
            - 0.0073 * x[:, 0] * x[:, 1]
            + 0.024 * x[:, 1] * x[:, 2]
            - 0.0118 * x[:, 1] * x[:, 3]
            - 0.0204 * x[:, 2] * x[:, 3]
            - 0.008 * x[:, 2] * x[:, 4]
            - 0.0241 * x[:, 1] ** 2
            + 0.0109 * x[:, 3] ** 2
        )



    objective_1 = ScalarObjective(name="the mass of the vehicle", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="acceleration-induced biomechanical damage of occupants", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the toe board intrusion in the 'offset-frontal crash'", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    x_1 = Variable("x_1", var_iv[0], 1.0, 3.0)
    x_2 = Variable("x_2", var_iv[1], 1.0, 3.0)
    x_3 = Variable("x_3", var_iv[2], 1.0, 3.0)
    x_4 = Variable("x_4", var_iv[3], 1.0, 3.0)
    x_5 = Variable("x_5", var_iv[4], 1.0, 3.0)

    variables = [x_1, x_2, x_3, x_4, x_5]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem