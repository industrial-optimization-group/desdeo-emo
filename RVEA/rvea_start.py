from deap import benchmarks, tools
# import optproblems as optp
import pygmo as pg
import pyRVEA as rv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pareto
import csv
import os

# (n,m) n samples & m objectives
s = 275  # initial population (50 105 120 126 132 112 156 90 275 for M-1 objectives)
n=12	#dimensions
m=10	#objectives
#n2=int(12/m) #number of points for reference vector
#n2=112	#number of reference vectors
method='Random'
generations=1000#1000,500
alpha=2.0
fr=0.1
FEmax=100000
p1=3 # 49 13  7  5  4  3  3  2  3
p2=2 # 0  0  0  0  1  2  2  2  2   for M-1 objectives
etac=30
etam=20
indpb=0 #

referencePoint = np.ones(m)*2

nrun=20
hyv=[]

Folder='DTLZ2_10'
if not os.path.exists(Folder):
    os.makedirs(Folder)
#The function evaluation
def fun(ind):
	udp = pg.problem(pg.dtlz(prob_id = 2, fdim =m, dim = n))  # , alpha=100 #pygmo benchmaks
	fitness_val=udp.fitness(ind)	
	#fitness_val=benchmarks.dtlz2(ind, m) #,alpha=100)	#deap benchmarks
	return(fitness_val)

for run in range(1, nrun+1):
	pop,sols,FE=rv.rvea(s,n,m,method,generations,alpha,fr,FEmax,p1,p2,etac,etam,indpb,fun)
	print("FE=")
	print(FE)
	print("Final population")
	print(pop)
	myFile = open(Folder+'\PoplationFinal_'+str(run)+'.csv', 'w')
	with myFile:
		writer = csv.writer(myFile)
		writer.writerows(pop)
	myFile2 = open(Folder+'\SolutionsFinal_'+str(run)+'.csv', 'w')
	with myFile2:
		writer = csv.writer(myFile2)
		writer.writerows(sols)
	#nondf=tools.sortNondominated(sols,len(sols),first_front_only=True)
	nondf=np.array(pareto.eps_sort(sols))
	print("Non-DF")
	print(nondf)
	hv = pg.hypervolume(nondf)
	hyv.append(hv.compute(referencePoint, hv_algo=pg.hvwfg()))
	print("Hypervolume:")
	print(hyv)
#	#if m==2:
#	#	plt.plot(nondf[:,0],nondf[:,1],'ro')
#	#	plt.title("DTLZ4")
#	#	plt.axis([0, 2, 0, 2])
#	#	plt.show()
#	#else:
#	#	fig = plt.figure()
#	#	ax = plt.axes(projection='3d')
#	#	plt.title("DTLZ6")
#	#	ax.scatter(nondf[:,0],nondf[:,1],nondf[:,2]);
#	#	plt.show()

print("Mean Hypervolume")
print(np.mean(hyv)/(2**m))
print(np.std(hyv)/(2**m))

print("Done!!!")



