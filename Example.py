"""
An example file demonstrating the use of pyrvea.

Visit the github/binder page for further information.
"""
from pyrvea.Population.Population import Population
from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.NSGAIII import NSGAIII
from optproblems import dtlz
import numpy as np


class newProblem(baseProblem):
    """New problem description."""

    def objectives(self, decision_variables):
        x = np.asarray(decision_variables)
        obj_func = sum(x ** 2)
        return obj_func
    #     return dtlz.DTLZ3(self.num_of_objectives, self.num_of_variables)(
    #         decision_variables
    #     )


name = "DTLZ3"
k = 10
numobj = 3
numconst = 0
numvar = numobj + k - 1
problem = newProblem(name, 2, 2, numconst)

lattice_resolution = 4
population_size = 105

pop = Population(problem)

pop.evolve(NSGAIII)

pop.non_dominated()
refpoint = 2
volume = 2 ** numobj
print(pop.hypervolume(refpoint) / volume)
