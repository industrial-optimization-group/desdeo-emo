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

from initializations import Individual, ReferenceVectors
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle, randint


def rvea(problem, parameters):
    """Run RVEA."""
    # Initializing Reference Vectors
    reference_vectors = ReferenceVectors(
        parameters.lattice_resolution, parameters.number_of_objectives
        )
    refV = reference_vectors.neighbouring_angles()

    # Initializing Population
    population = [
        Individual(problem) for i in range(parameters.population_size)
        ]
    # Clever, but foolish. This won't work. :(
    population_fitness = [individual.evaluate() for individual in population]
    print('Initiating RVEA generations')
    for gen_count in range(parameters.generations):
        if gen_count % 10 == 0:
            print('Processing generation number %d\n', gen_count)
        # Random mating and evaluation
        offspring = create_offspring(population)
        offspring_fitness = [individual.evaluate() for individual in offspring]
        population = population + offspring
        population_fitness = population_fitness + offspring_fitness
        # APD Based selection
        penalty_factor = ((gen_count/parameters.generations) **
                          parameters.alpha)*problem.num_of_objectives
        select = APD_select(population_fitness, reference_vectors,
                            penalty_factor, refV)
        population_new = list(population(i) for i in select)
        population = population_new
        population_fitness_new = list(population_fitness(i) for i in select)
        population_fitness = population_fitness_new
        # Reference Vector Adaptation
        if ((gen_count) % ceil(parameters.generations *
                               parameters.refV_adapt_frequency)) == 0:
            zmax = max(population_fitness)
            zmin = min(population_fitness)
            reference_vectors.adapt(zmax, zmin)
            refV = reference_vectors.neighbouring_angles()
            plt.plot(list(x[0] for x in population_fitness),
                     list(x[1] for x in population_fitness))
            plt.show()


def APD_select(fitness: list, vectors: ReferenceVectors,
               penalty_factor: float, refV: np.ndarray):
    """Select individuals for mating on basis of Angle penalized distance.

    Returns a list of indices of the selected individuals.
    """
    fitness = np.asarray(fitness)
    # CV = fitness[:, -1]
    fitness = fitness[:, 0:-1]
    # Normalization - There may be problems here
    fmin = np.amin(fitness, axis=0)
    fitness = fitness - fmin
    fitness_norm = np.linalg.norm(fitness, axis=1)
    fitness_norm = np.repeat(fitness_norm, len(fitness[0, :])).reshape(
            len(fitness), len(fitness[0, :]))
    normalized_fitness = np.divide(fitness, fitness_norm)
    cosine = np.dot(normalized_fitness, np.transpose(vectors.values))
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
    selection = np.array([])
    # Selection
    for i in range(0, len(vectors.values)):
        sub_population_index = np.where(assigned_vectors == i)
        sub_population_fitness = fitness[sub_population_index]
        if len(sub_population_fitness > 0):
            # APD Calculation
            angles = theta[sub_population_index, i]
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
            selx = np.asarray(sub_population_index)[minidx]
            if selection.shape[0] == 0:
                selection = np.hstack((selection, np.transpose(selx[0])))
            else:
                selection = np.vstack((selection, np.transpose(selx[0])))
    return(selection)


def create_offspring(population: list, problem, parameters) -> list:
    """Conduct crossover and mutation over the population.

    Return offspring population
    """
    population_size = len(population)
    if population_size % 2 == 1:
        mating_pop = population + population[randint(0, population_size-1)]
        population_size = population_size + 1
    else:
        mating_pop = population
    shuffled_ids = range(population_size)
    shuffle(shuffled_ids)
    offspring = []
    for i in range(0, population_size, 2):
        individual1 = mating_pop[i]
        individual2 = mating_pop[i+1]
        child1, child2 = Individual.mate(individual1, individual2,
                                         problem, parameters)
        offspring = offspring + [child1] + [child2]
    return offspring
    pass
