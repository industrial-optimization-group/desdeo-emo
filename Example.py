"""
An example file demonstrating the use of pyRVEA.

Visit the github/binder page for further information.
"""
from pyRVEA.allclasses import Problem, Parameters, Population
from optproblems import dtlz


class newProblem(Problem):
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

lattice_resolution = 13
population_size = 105

parameters = Parameters(
    population_size, lattice_resolution, algorithm_name="RVEA", interact=False
)

pop = Population(problem, parameters)

newpop = pop.evolve(problem, parameters)

newpop.non_dominated()
refpoint = 2
volume = 2 ** numobj
print(newpop.hypervolume(refpoint) / volume)
