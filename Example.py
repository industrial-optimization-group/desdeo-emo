"""
An example file demonstrating the use of pyrvea.

Visit the github/binder page for further information.
"""
from pyrvea.Population.Population import Population
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
from optproblems import dtlz


class newProblem(BaseProblem):
    """New problem description."""

    def objectives(self, decision_variables):
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

pop = Population(
    problem,
    crossover_type="simulated_binary_crossover",
    mutation_type="bounded_polynomial_mutation",
    plotting=True
)

pop.evolve(RVEA)

pop.non_dominated()
refpoint = 2
volume = 2 ** numobj
print(pop.hypervolume(refpoint) / volume)
