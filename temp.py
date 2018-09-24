from initializations import Parameters, Problem
from RVEA import rvea


name = 'ZDT1'
numvar = 30
uplim = 1
lowlim = 0
numobj = 2
numconst = 0

problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)

population_size = 100
lattice_resolution = 10
generations = 1000

parameters = Parameters(population_size, lattice_resolution, generations)

rvea(problem, parameters)