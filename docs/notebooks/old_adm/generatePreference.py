# from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
import numpy as np
import baseADM


def generateRP4learning(base: baseADM):

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    # Assigment of the solutions to the vectors
    assigned_vectors = base.assigned_vectors

    # Find the vector which has a minimum number of assigned solutions
    number_assigned = np.bincount(assigned_vectors)
    min_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.min(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == min_assigned_vector[0]))
        # If there are multiple vectors which have the minimum number of solutions, first one's index is used
    )
    # Assigned solutions to the vector which has a minimum number of solutions
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[min_assigned_vector[0]]
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point


def get_max_assigned_vector(assigned_vectors):

    number_assigned = np.bincount(assigned_vectors)
    max_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.max(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    return max_assigned_vector


def generateRP4decision(base: baseADM, max_assigned_vector):

    assigned_vectors = base.assigned_vectors

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == max_assigned_vector))
    )
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[max_assigned_vector]
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point
