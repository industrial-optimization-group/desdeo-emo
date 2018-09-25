from initializations import Parameters, Problem
from RVEA import rvea


name = 'DTLZ6'

uplim = 1
lowlim = 0
numobj = 8
k = 10
numvar = numobj + k - 1
numconst = 0

problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)

population_size = 100
lattice_resolution = 3
generations = 500

parameters = Parameters(population_size, lattice_resolution, generations)

rvea(problem, parameters)

