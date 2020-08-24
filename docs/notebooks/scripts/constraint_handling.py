import numpy as np
import pandas as pd

from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Variable import variable_builder
from desdeo_problem.Problem import MOProblem
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_

from desdeo_problem.Constraint import ScalarConstraint

# ## MOP-C4 Tanaka from Coello's book
#
# The problem is wrong in the book, look up literature.


def f_1(x):
    return x[:, 0]


def f_2(x):
    return x[:, 1]


def c_1(x, y):
    a = 0.1
    b = 16
    return (
        -x[:, 0] ** 2 - x[:, 1] ** 2 + 1 + a * np.cos(b * np.arctan(x[:, 0] / x[:, 1]))
    )


def c_2(x, y):
    return -0.5 + (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2


list_vars = variable_builder(
    ["x", "y"], initial_values=[0, 0], lower_bounds=[0, 0], upper_bounds=[np.pi, np.pi]
)


f1 = _ScalarObjective(name="f1", evaluator=f_1)
f2 = _ScalarObjective(name="f2", evaluator=f_2)


c1 = ScalarConstraint("c1", 2, 2, c_1)
c2 = ScalarConstraint("c2", 2, 2, c_2)


problem = MOProblem(variables=list_vars, objectives=[f1, f2], constraints=[c1, c2])

evolver = RVEA(problem)
"""
figure = animate_init_(evolver.population.objectives, filename="MOPC4.html")
"""
while evolver.continue_evolution():
    evolver.iterate()
    """figure = animate_next_(
        evolver.population.objectives,
        figure,
        filename="MOPC4.html",
        generation=evolver._iteration_counter,
    )
"""
