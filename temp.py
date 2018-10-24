from initializations import Parameters, Problem
from RVEA import rvea
import os
names = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
k = [5, 10, 10, 10, 10, 10, 10]
uplim = 1
lowlim = 0
numobjs = [2, 3, 6, 8, 10, 12]

lattice_resolution = [100, 20, 5, 4, 3, 3]
population_size = 500
generations = 1000
# numvar = 30
numconst = 0
for name, K in zip(names, k):
    for numobj, lr in zip(numobjs, lattice_resolution):
        numvar = numobj + K - 1
        problem = Problem(name, numvar, uplim, lowlim, numobj, numconst)
        parameters = Parameters(population_size, lr, generations)
        try:
            [fnd, fpg, timeelapsed] = rvea(problem, parameters)
            foldername = '/results/'
            filename1 = name + '-' + str(numobj) + 'nd.txt'
            filename2 = name + '-' + str(numobj) + 'everygen.txt'
            fullfilename1 = os.getcwd() + foldername + filename1
            fullfilename2 = os.getcwd() + foldername + filename2
            os.makedirs(os.path.dirname(fullfilename1), exist_ok=True)
            os.makedirs(os.path.dirname(fullfilename2), exist_ok=True)
            with open(fullfilename1, 'w') as f:
                for item in fnd:
                        f.write("%s\n" % item)
            with open(fullfilename2, 'w') as f:
                for item in fpg:
                    for smallitem in item:
                        f.write("%s\n" % smallitem)
        except Exception as err:
            with open("Errorlog.txt", 'a') as f:
                f.write("Error in " + name + " with " + str(numobj) +
                        "objectives\n")
                f.write("{} \n".format(err))
