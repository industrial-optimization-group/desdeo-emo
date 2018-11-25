# %%
from initializations import Problem, Parameters, Population
# %%
name = 'DTLZ3'
k = 10
uplim = 1
lowlim = 0
numobj = 3
numconst = 0
numvar = numobj + k - 1
# %%
lattice_resolution = 13
population_size = 500

problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)

parameters = Parameters(population_size, lattice_resolution)

pop = Population(problem, parameters)
# %%
newpop = pop.evolve(problem, parameters)
newpop.non_dom()