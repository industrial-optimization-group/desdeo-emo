"""
An example file demonstrating the use of pyRVEA.

Visit the github/binder page for further information.
"""
from pyRVEA.Population.Population import Population
from pyRVEA.Problem.baseProblem import baseProblem
from pyRVEA.EAs.RVEA import RVEA
from optproblems import dtlz


class newProblem(baseProblem):
    """New problem description."""

    def objectives(self, decision_variables):
        """Return objective value."""
        return dtlz.DTLZ3(self.num_of_objectives, self.num_of_variables)(
            decision_variables
        )


name = "DTLZ3"
k = 10
numobj = 3
numconst = 0
numvar = numobj + k - 1
problem = newProblem(name, numvar, numobj, numconst)

lattice_resolution = 4
population_size = 105

pop = Population(problem)

newpop = pop.evolve(RVEA)

newpop.non_dominated()
refpoint = 2
volume = 2 ** numobj
print(newpop.hypervolume(refpoint) / volume)
