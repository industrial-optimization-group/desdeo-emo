from initializations import Parameters, Problem
from RVEA import rvea
import numpy as np
from pygmo import hypervolume as hv


name = 'DTLZ3'
k = 10
uplim = 1
lowlim = 0
numobj = 6

lattice_resolution = 4
population_size = 500
generations = 2000


numconst = 0
numvar = numobj + k - 1
problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)
parameters = Parameters(population_size, lattice_resolution, generations)
[fnd, timeelapsed] = rvea(problem, parameters)
fnd = np.asarray(fnd)[:, 0:-1]
check = 0
for index in range(0, fnd.shape[0]):
    if sum(fnd[index] > 2) == 0:
        if check == 0:
            pareto = fnd[index]
            check += 1
        else:
            pareto = np.vstack((pareto, fnd[index]))

hyp = hv(pareto)
print(hyp.compute([2.0]*6))
print('Finished')

# name = 'ZDT1'
# uplim = 1
# lowlim = 0
# numobj = 2
# lattice_resolution = 100
# population_size = 500
# generations = 500
# numconst = 0
# numvar = 30
# problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)
# parameters = Parameters(population_size, lattice_resolution, generations)
# [fnd, fpg, timeelapsed] = rvea(problem, parameters)