from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase

import numpy as np

def river_pollution_problem(five_obj: bool = True, var_iv: np.array = np.array([0.5, 0.5])) -> MOProblem:
    """The river pollution problem with 4 or 5 objectives.

    NARULA, S. C. & WEISTROFFER, H. R. (1989). A flexible method for 
    nonlinear multicriteria decisionmaking problems. IEEE transactions on 
    systems, man, and cybernetics, 19(4), 883-887.

    Arguments:
        five_obj (bool): If true utilize five objectives version and four objectives
            version if false. Default is true. 
        var_iv (np.array): Optional, initial variable values. Must be between 0.3 and 1.0.
            Defaults are 0.5 and 0.5. 

    Returns:
        MOProblem: a problem object.
    """
    
    if np.any(1 < var_iv) or np.any(var_iv < 0.3):

        raise ValueError("Initial variable values need to be between lower and upper bounds")

    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -4.07 - 2.27*x[:, 0]

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -2.60 - 0.03*x[:, 0] - 0.02*x[:, 1] - 0.01 / (1.39 - x[:, 0]**2) - 0.30 / (1.39 - x[:, 1]**2)

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -8.21 + 0.71 / (1.09 - x[:, 0]**2)

    def f_4(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -0.96 + 0.96 / (1.09 - x[:, 1]**2)
    
    objective_1 = ScalarObjective(name="the DO level in the city", evaluator=f_1, maximize=[True])
    objective_2 = ScalarObjective(name="the DO level at the municipality border", evaluator=f_2, maximize=[True])
    objective_3 = ScalarObjective(name="the percent return on investment at the fishery", evaluator=f_3, maximize=[True])
    objective_4 = ScalarObjective(name="the addition to the tax rate of city", evaluator=f_4, maximize=[True])
    

    if five_obj:
        def f_5(x: np.ndarray) -> np.ndarray:
            return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

        objective_5 = ScalarObjective(name="BOD removed form the water close to the ideal value of 0.65", evaluator=f_5, maximize=[False])
        objectives = [objective_1, objective_2, objective_3, objective_4, objective_5]
    else:
        # If five_obj is false, then problem is with 4 objectives. 
        objectives = [objective_1, objective_2, objective_3, objective_4]

    x_1 = Variable("the proportionate amount of BOD removed from water at the fishery", var_iv[0], 0.3, 1.0)
    x_2 = Variable("the proportionate amount of BOD removed from water at the city", var_iv[1], 0.3, 1.0)

    variables = [x_1, x_2]
    
    problem = MOProblem(variables=variables, objectives=objectives)

    return problem
