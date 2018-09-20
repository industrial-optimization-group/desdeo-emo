from random import randint
from pyDOE import lhs
from deap import tools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import comb

def rvea(s,n,m,method,generations,alpha,fr,FEmax,p1,p2,etac,etam,indpb,fun):
	FE=0
	#Create vectors and normalize
	u=F_weights(m,p1,p2)
	vinit=vi_transform(u)
	v=vinit
	refV=ref_v_norm(v)
	#Generating initial population
	pop = _lhs_gen(s,n,method) 
	fitness_pop=obj_calc(pop, m, fun)
	print("Generations:")
	for curr_gen in range(0,generations):
		print(curr_gen)	
		#Evolve
		offspring=mate(pop,refV.shape[0],etac,etam,lower=np.zeros(pop.shape[1]).tolist(),upper=np.ones(pop.shape[1]).tolist(),indpb=indpb)
		FE=FE+offspring.shape[0]
		fitness_off=obj_calc(offspring, m, fun)
		pop=np.concatenate((pop,offspring),axis=0)
		fitness_pop=np.concatenate((fitness_pop,fitness_off),axis=0)
		theta0= ((curr_gen/generations)**alpha)*m
		selection=apd_select(v,fitness_pop,refV,theta0,pop)
		selloc=tuple(np.ndarray.tolist(np.transpose(selection))[0])
		pop=np.take(pop,selloc,axis=0)
		fitness_pop=np.take(fitness_pop,selloc,axis=0)
		#Reference vector adaptation
		if ((curr_gen) % math.ceil(generations*fr))==0:
			print("Adapting Reference Vectors...")
			Zmin=np.amin(fitness_pop,axis=0)	 
			Zmax=np.amax(fitness_pop,axis=0)
			n3=vinit.shape[0]
			v=np.multiply(vinit,np.tile(np.subtract(Zmax,Zmin),(n3,1)))
			v=vi_transform(v)
			refV=ref_v_norm(v)
	sols=np.array(obj_calc(pop, m, fun))
	return(pop,sols,FE)

# LHS initial population (n,m) n samples & m objectives
def _lhs_gen(n,m,method):
	if method=='LHS':
		return(np.array(lhs(m,samples=n)))
	else:	
		return(np.array(np.random.rand(n,m)))
	

# Assigning vectors using simplex lattice

def F_weights(M,p1,p2):
	max=comb(p1+M-1, M-1, exact=True)
	wt=np.asarray(list(simplex_lattice(M-1,max,p1+1)))
	wt=wt/wt[0][wt.shape[1]-1]
	if p2>0:
		max=comb(p2+M-1, M-1, exact=True)
		wt2=np.asarray(list(simplex_lattice(M-1,max,p2+1)))
		wt2=wt2/wt2[0][wt2.shape[1]-1]
		wt2=np.dot(wt2,0.5)+(1-0.5)/(M)
		wt=np.vstack((wt,wt2))
	return(wt)
	
def simplex_lattice(num_dims, samples_per_dim, max_):
	if num_dims == 0:
		yield np.array([(max_ - 1) / (samples_per_dim - 1)])
	else:
		for i in range(max_):
			for rest in simplex_lattice(num_dims - 1, samples_per_dim, max_ - i):
				yield np.concatenate((np.array([i]) / (samples_per_dim - 1),rest))
	

# Calculate objective values
def obj_calc(pop,m,fun):
	obj_vec=[]
	for i in range(0,len(pop)):	
		obj_val=fun(pop[i,:])		
		obj_vec.append(obj_val)
	return(obj_vec)
	
# Reference vector normalization
def vi_transform(u):
	u_norm=np.linalg.norm(u,axis=1)
	u_norm=np.repeat(u_norm, len(u[0,:])).reshape(len(u),len(u[0,:]))
	v=np.divide(u,u_norm) # u/u_norm forming the vectors v
	return(v)

# Neighbouring angle calcuation for angle normalization
def ref_v_norm(v):
	cosvv=np.dot(v, v.transpose())
	cosvv.sort(axis=1)
	cosvv=np.flip(cosvv,1)
	acosvv=np.arccos(cosvv[:,1])
	return(acosvv)

# Objective value translation
def obj_val_translate(f):
	f=np.asarray(f)
	fmin=np.amin(f, axis=0)
	return(f-fmin)

# APD based selection
def apd_select(v,f1,refV,theta0,pop):
	f=obj_val_translate(f1)
	f_norm=np.linalg.norm(f,axis=1)
	if len(f_norm[np.where(f_norm==0)])>0:
		print("Whaattttt!")
		print(np.where(f_norm==0))
	
	f_norm=np.repeat(f_norm, len(f[0,:])).reshape(len(f),len(f[0,:]))
	f_vec=np.divide(f,f_norm)
	cos_theta=np.dot(f_vec,np.transpose(v))
	
	theta = np.array([])

	
	for i in range(0,len(cos_theta)):
		#print(cos_theta[i,:])
		thetatemp=np.arccos(cos_theta[i,:])
		if i==0:
			theta=np.hstack((theta,thetatemp))
		else:
			theta=np.vstack((theta,thetatemp))
	arg_max_index=np.argmax(cos_theta, axis=1)
	selection = np.array([])
	for i in range(0,len(v)):
		sub=np.where(arg_max_index==i)
		sub_fun_val=f[sub]
		if len(sub_fun_val)>0:
			# Angle penalized distance calculation
			subacos = theta[sub,i]
			subacos = np.divide(subacos,refV[i]) # Angle normalization
			D1=np.sqrt(np.sum(np.power(sub_fun_val,2),axis=1))	#Eculidean distance
			D = np.multiply(np.transpose(D1),(1+np.dot(theta0,subacos))) #APD
			minidx= np.where(D==np.amin(D))
			selx=np.asarray(sub)[minidx]
			if selection.shape[0]==0:
				selection=np.hstack((selection,np.transpose(selx[0])))
			else:
				selection=np.vstack((selection,np.transpose(selx[0])))
	return(selection)

#SBX Crossover and Polynomial Mutation
def mate(pop,max_off,etac,etam,lower,upper,indpb):
	if indpb==0:
		indpb=1/pop.shape[1]
	pop=np.random.permutation(pop)
	if max_off<1 or pop.shape[0]<max_off:
		max_off=pop.shape[0]
	child=np.empty([pop.shape[0]+2,pop.shape[1]])
	for i in range(0,(math.floor(pop.shape[0]/2)+1)):
		same=True
		while same:
			pos1=(randint(0,pop.shape[0]-1))
			pos2=(randint(0,pop.shape[0]-1))
			ind1=np.ndarray.tolist(pop[pos1,:])
			ind2=np.ndarray.tolist(pop[pos2,:])
			oldind1=np.ndarray.tolist(pop[pos1,:])
			oldind2=np.ndarray.tolist(pop[pos2,:])
			childcx=np.asarray(tools.cxSimulatedBinaryBounded(ind1, ind2, etac, lower, upper))
			same=False
			if np.array_equal(np.around(oldind1,decimals=4),np.around(ind1,decimals=4)) or np.array_equal(np.around(oldind1,decimals=4),np.around(ind2,decimals=4)) or np.array_equal(np.around(oldind2,decimals=4),np.around(ind1,decimals=4)) or np.array_equal(np.around(oldind2,decimals=4),np.around(ind2,decimals=4)):
				same = True
		child[2*i]=np.asarray(tools.mutPolynomialBounded(ind1,etam,lower,upper,indpb))
		child[2*i+1]=np.asarray(tools.mutPolynomialBounded(ind2,etam,lower,upper,indpb))
	child=child[0:max_off,:]
	return(child)	




