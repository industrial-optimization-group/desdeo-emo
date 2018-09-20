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

def rvea(problem, parameters):
    """Run RVEA."""
    # Initializing Reference Vectors
    reference_vectors = ReferenceVectors(
        parameters.lattice_resolution, parameters.number_of_objectives
        )
    reference_vectors.normalize()
    refV = reference_vectors.neighbouring_angles()

    # Initializing Population
    population = [
        Individual(problem) for i in range(parameters.population_size)
        ]
    population_fitness = [individual.evaluate() for individual in population]
    print('Initiating RVEA generations')
    for gen_count in range(parameters.generations):
        if gen_count % 10 == 0:
            print('Processing generation number %d\n', gen_count)
        # Random mating and evaluation
        offspring = mate(population)
        offspring_fitness = [individual.evaluate() for individual in offspring]
        population = population + offspring
        population_fitness = population_fitness + offspring_fitness
        # APD Based selection
        theta0 = ((gen_count/parameters.generations) **
                  parameters.alpha)*problem.num_of_objectives
        select = APD_select(population_fitness, reference_vectors, theta0,
                            refV)
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
            reference_vectors.normalize()
            refV = reference_vectors.neighbouring_angles()
            plt.plot(list(x[0] for x in population_fitness),
                     list(x[1] for x in population_fitness))
            plt.show()


def APD_select(fitness, vectors, theta0, refV):
    fitness = np.asarray(fitness)
    CV = fitness[:, -1]
    fitness = fitness[:, 0:-1]
    fmin = np.amin(fitness, axis=0)
    fitness = fitness - fmin
    fitness_norm = np.linalg.norm(fitness, axis=1)
