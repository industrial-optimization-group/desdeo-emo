import numpy as np
from warnings import warn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrvea.allclasses import ReferenceVectors


def APD_select(
    fitness: list,
    vectors: "ReferenceVectors",
    penalty_factor: float,
    ideal: list = None,
):
    """Select individuals for mating on basis of Angle penalized distance.

    Args:
        fitness (list): Fitness of the current population.

        vectors (ReferenceVectors): Class containing reference vectors.

        penalty_factor (float): Multiplier of angular deviation from Reference
            vectors. See RVEA paper for details.

        ideal (list): ideal point for the population.
            Uses the min fitness value if None.

    Returns:
        [type]: A list of indices of the selected individuals.
    """
    refV = vectors.neighbouring_angles_current
    # Normalization - There may be problems here
    if ideal is not None:
        fmin = ideal
    else:
        fmin = np.amin(fitness, axis=0)
    translated_fitness = fitness - fmin
    fitness_norm = np.linalg.norm(translated_fitness, axis=1)
    fitness_norm = np.repeat(fitness_norm, len(translated_fitness[0, :])).reshape(
        len(fitness), len(fitness[0, :])
    )
    normalized_fitness = np.divide(translated_fitness, fitness_norm)  # Checked, works.
    cosine = np.dot(normalized_fitness, np.transpose(vectors.values))
    if cosine[np.where(cosine > 1)].size:
        warn("RVEA.py line 60 cosine larger than 1 decreased to 1")
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        warn("RVEA.py line 64 cosine smaller than 0 increased to 0")
        cosine[np.where(cosine < 0)] = 0
    # Calculation of angles between reference vectors and solutions
    theta = np.arccos(cosine)
    # Reference vector asub_population_indexssignment
    assigned_vectors = np.argmax(cosine, axis=1)
    selection = np.array([], dtype=int)
    # Selection
    for i in range(0, len(vectors.values)):
        sub_population_index = np.atleast_1d(
            np.squeeze(np.where(assigned_vectors == i))
        )
        sub_population_fitness = translated_fitness[sub_population_index]
        if len(sub_population_fitness > 0):
            # APD Calculation
            angles = theta[sub_population_index, i]
            angles = np.divide(angles, refV[i])  # This is correct.
            # You have done this calculation before. Check with fitness_norm
            # Remove this horrible line
            sub_pop_fitness_magnitude = np.sqrt(
                np.sum(np.power(sub_population_fitness, 2), axis=1)
            )
            apd = np.multiply(
                np.transpose(sub_pop_fitness_magnitude),
                (1 + np.dot(penalty_factor, angles)),
            )
            minidx = np.where(apd == np.amin(apd))
            selx = sub_population_index[minidx]
            if selection.shape[0] == 0:
                selection = np.hstack((selection, np.transpose(selx[0])))
            else:
                selection = np.vstack((selection, np.transpose(selx[0])))
    return selection.squeeze()
