"""The constrained verstion reference vector guided evolutionary algorithm.

See the details of RVEA in the following paper

R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
A Reference Vector Guided Evolutionary Algorithm for Many-objective
Optimization, IEEE Transactions on Evolutionary Computation, 2016

The source code of cRVEA is implemented by Bhupinder Saini

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi
Project researcher at University of Jyväskylä.
"""

from math import ceil
import numpy as np
from random import shuffle, randint
from pareto import eps_sort
from progress.bar import IncrementalBar as Bar
from time import time
from warnings import warn
from scipy.special import comb
from itertools import combinations


class ReferenceVectors():
    """Class object for reference vectors."""

    def __init__(self, lattice_resolution: int, number_of_objectives):
        """Create a simplex lattice."""
        number_of_vectors = comb(
            lattice_resolution + number_of_objectives - 1,
            number_of_objectives - 1, exact=True)
        temp1 = range(1, number_of_objectives + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, number_of_objectives-1)))
        temp2 = np.array([range(number_of_objectives-1)]*number_of_vectors)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_vectors, number_of_objectives), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, number_of_objectives-1):
            weight[:, i] = temp[:, i] - temp[:, i-1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        self.values = weight/lattice_resolution
        self.initial_values = self.values
        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_vectors = number_of_vectors
        self.normalize()

    def normalize(self):
        """Normalize the reference vectors."""
        norm = np.linalg.norm(self.values, axis=1)
        norm = np.repeat(norm, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives)
        self.values = np.divide(self.values, norm)

    def neighbouring_angles(self) -> np.ndarray:
        """Calculate neighbouring angles for normalization."""
        cosvv = np.dot(self.values, self.values.transpose())
        cosvv.sort(axis=1)
        cosvv = np.flip(cosvv, 1)
        acosvv = np.arccos(cosvv[:, 1])
        return(acosvv)

    def adapt(self, max_val, min_val):
        """Adapt reference vectors."""
        self.values = np.multiply(self.initial_values,
                                  np.tile(np.subtract(max_val, min_val),
                                          (self.number_of_vectors, 1)))
        self.normalize()


def rvea(population, problem, parameters):
    """Run RVEA."""
    start_time = time()
    # Initializing Reference Vectors
    reference_vectors = ReferenceVectors(
        parameters['lattice_resolution'], problem.num_of_objectives)
    refV = reference_vectors.neighbouring_angles()
    print('Running RVEA generations\n')
    # setup toolbar
    bar = Bar('Processing', max=parameters['generations'])
    # setup plots
    ploton = parameters['ploton']

    for gen_count in range(parameters['generations']):
        # Random mating and evaluation
        offspring = population.mate()
        population.add(offspring, problem)
        # APD Based selection
        penalty_factor = ((gen_count/parameters['generations']) **
                          parameters['Alpha'])*problem.num_of_objectives
        select = APD_select(population.fitness, reference_vectors,
                            penalty_factor, refV)
        population.keep(select)
        # Reference Vector Adaptation
        if ((gen_count) % ceil(parameters['generations'] *
                               parameters['refV_adapt_frequency'])) == 0:
            zmax = np.amax(np.asarray(population.fitness), axis=0)
            zmin = np.amin(np.asarray(population.fitness), axis=0)
            # print('Processing generation number', gen_count, '\n')
            reference_vectors.adapt(zmax, zmin)
            refV = reference_vectors.neighbouring_angles()
            # plotting
            if ploton:
                population.plot_objectives()
        bar.next()
    bar.finish()
    time_elapsed = time() - start_time
    # plotting
    if ploton:
        population.plot_objectives()
    return(population, time_elapsed)


def APD_select(fitness: list, vectors: ReferenceVectors,
               penalty_factor: float, refV: np.ndarray):
    """Select individuals for mating on basis of Angle penalized distance.

    Returns a list of indices of the selected individuals.
    """
    # Normalization - There may be problems here
    fmin = np.amin(fitness, axis=0)
    fitness = fitness - fmin
    fitness_norm = np.linalg.norm(fitness, axis=1)
    fitness_norm = np.repeat(fitness_norm, len(fitness[0, :])).reshape(
            len(fitness), len(fitness[0, :]))
    normalized_fitness = np.divide(fitness, fitness_norm)
    cosine = np.dot(normalized_fitness, np.transpose(vectors.values))
    if cosine[np.where(cosine > 1)].size:
        warn('RVEA.py line 103 cosine larger than 1 decreased to 1:',
             cosine[np.where(cosine > 1)])
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        warn('RVEA.py line 103 cosine smaller than 0 decreased to 0:',
             cosine[np.where(cosine < 0)])
        cosine[np.where(cosine < 0)] = 0
    theta = np.array([])
    # Calculation of angles between reference vectors and solutions
    for i in range(0, len(cosine)):
        thetatemp = np.arccos(cosine[i, :])
        # Shenanigans to keep the correct shape. Find a better way to do this?
        if i == 0:
            theta = np.hstack((theta, thetatemp))
        else:
            theta = np.vstack((theta, thetatemp))
    # Better way? - theta = np.arccos(cosine)
    # Reference vector assignment
    assigned_vectors = np.argmax(cosine, axis=1)
    selection = np.array([], dtype=int)
    # Selection
    for i in range(0, len(vectors.values)):
        sub_population_index = np.where(assigned_vectors == i)
        sub_population_fitness = fitness[sub_population_index]
        if len(sub_population_fitness > 0):
            # APD Calculation
            angles = theta[sub_population_index[0], i]
            angles = np.divide(angles, refV[i])  # min(refV[i])?
            # You have done this calculation before. Check with fitness_norm
            # Remove this horrible line
            sub_pop_fitness_magnitude = np.sqrt(
                                            np.sum(
                                                np.power(
                                                    sub_population_fitness,
                                                    2), axis=1))
            apd = np.multiply(np.transpose(sub_pop_fitness_magnitude),
                              (1 + np.dot(penalty_factor, angles)))
            minidx = np.where(apd == np.amin(apd))
            selx = np.asarray(sub_population_index)[0][minidx]
            if selection.shape[0] == 0:
                selection = np.hstack((selection, np.transpose(selx[0])))
            else:
                selection = np.vstack((selection, np.transpose(selx[0])))
    return(selection.squeeze())


