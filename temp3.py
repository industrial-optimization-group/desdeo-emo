from initializations import Parameters, Problem
from RVEA import rvea
import numpy as np
from pygmo import hypervolume as hv
import pickle as pk


name = 'DTLZ3'
k = 10
uplim = 1
lowlim = 0
numobj = 3

lattice_resolution = 13
population_size = 500
generations = [1000, 1500, 2000, 2500, 3000]
hypervolume = {'1000': [],
               '1500': [],
               '2000': [],
               '2500': [],
               '3000': []}

numconst = 0
numvar = numobj + k - 1

problem = Problem(name,
                  numvar,
                  uplim,
                  lowlim,
                  numobj,
                  numconst)

for run_index in range(0,10):
    for generation in generations:
        parameters = Parameters(population_size,
                                lattice_resolution,
                                generation)
        [fnd, timeelapsed] = rvea(problem, parameters)
        fnd = np.asarray(fnd)[:, 0:-1]
        check = 0
        pareto = []
        for index in range(0, fnd.shape[0]):
            if sum(fnd[index] > 2) == 0:
                if check == 0:
                    pareto = fnd[index]
                    check += 1
                else:
                    pareto = np.vstack((pareto, fnd[index]))
        if list(pareto):
            hyp = hv(pareto)
            hyperv = (hyp.compute([2.0]*3))/8
            print(generation, hyperv)
            hypervolume[str(generation)].append(hyperv)
        else:
            hypervolume[str(generation)].append(0)
            print(generation, 0)
pk.dump(hypervolume, open("hypervolume.p", "wb"))
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