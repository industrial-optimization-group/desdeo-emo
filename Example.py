from allclasses import Problem, Parameters, Population
from optproblems import dtlz

class newProblem(Problem):
    def objectives(self, decision_variables):
        return(dtlz.DTLZ3(3,12)(decision_variables))


name = 'DTLZ3'
k = 10
numobj = 3
numconst = 0
numvar = numobj + k - 1
problem = newProblem(name, numvar, numobj, numconst)

lattice_resolution = 13
population_size = 500

parameters = Parameters(population_size, lattice_resolution, algorithm_name='RVEA')

pop = Population(problem, parameters)

newpop = pop.evolve(problem, parameters)

newpop.non_dominated()
refpoint = 2
volume = 2 ** numobj
print(newpop.hypervolume(refpoint)/volume)
