"""The python version reference vector guided evolutionary algorithm.

See the details of RVEA in the following paper

R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
A Reference Vector Guided Evolutionary Algorithm for Many-objective
Optimization, IEEE Transactions on Evolutionary Computation, 2016

The source code of cRVEA is implemented by Bhupinder Saini

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi
Project researcher at University of Jyväskylä.
"""

import numpy as np
from warnings import warn


def rvea(population,
         problem,
         parameters,
         reference_vectors,
         progressbar):
    """
    Run RVEA.

    This only conducts reproduction and selection.
    Reference vector adaptation should be done outside.
    """
    refV = reference_vectors.neighbouring_angles()
    progress = progressbar(range(parameters['generations']),
                           desc='Generations',
                           leave=False)
    for gen_count in progress:
        offspring = population.mate()
        population.add(offspring, problem)
        # APD Based selection
        penalty_factor = ((gen_count/parameters['generations']) **
                          parameters['Alpha'])*problem.num_of_objectives
        select = APD_select(population.fitness, reference_vectors,
                            penalty_factor, refV)
        population.keep(select)
    progress.close()
    return(population)


def APD_select(fitness: list, vectors: 'ReferenceVectors',
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
        warn('RVEA.py line 60 cosine larger than 1 decreased to 1:',
             cosine[np.where(cosine > 1)])
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        warn('RVEA.py line 64 cosine smaller than 0 decreased to 0:',
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
