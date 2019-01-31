"""Testing code."""

import numpy as np
from pyRVEA.EAs.RVEA import rvea

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population
    from pyRVEA.Problem.baseProblem import baseProblem
    from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors


class Parameters:
    """This object contains the parameters necessary for evolution."""

    def __init__(
        self,
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = True,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 10,
        Alpha: float = 2,
        plotting: bool = True,
        algorithm_name="RVEA",
        *args
    ):
        """Initialize the parameters class."""
        self.algorithm_name = algorithm_name
        if algorithm_name == "RVEA":
            rveaparams = {
                "population_size": population_size,
                "lattice_resolution": lattice_resolution,
                "algorithm": rvea,
                "interact": interact,
                "a_priori": a_priori_preference,
                "generations": generations_per_iteration,
                "iterations": iterations,
                "Alpha": Alpha,
                "ploton": plotting,
            }
        self.params = rveaparams


def interrupt_evolution(
    reference_vectors: "ReferenceVectors",
    population: "Population",
    problem: "baseProblem" = None,
    parameters: "Parameters" = None,
):
    """Perform operations while optimization is interrupted.

    Currently supported: Adaptaion of reference vectors with or without preference info.

    Parameters
    ----------
    reference_vectors: ReferenceVectors Object

    population: A Population Object
    problem: Object of the class Problem or derived from class Problem.

    """
    if parameters.algorithm_name == "RVEA":
        if parameters.params["interact"] or parameters.params["a_priori"]:
            # refpoint = np.mean(population.fitness, axis=0)
            ideal = np.amin(population.fitness, axis=0)
            nadir = np.amax(population.fitness, axis=0)
            refpoint = np.zeros_like(ideal)
            print("Ideal vector is ", ideal)
            print("Nadir vector is ", nadir)
            for index in range(len(refpoint)):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    print("Nadir value = ", nadir[index])
                    pref_val = float(
                        input("Please input a value between ideal and nadir: ")
                    )
                    if pref_val > ideal[index] and pref_val < nadir[index]:
                        refpoint[index] = pref_val
                        break
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            reference_vectors.iteractive_adapt_1(refpoint)
            reference_vectors.add_edge_vectors()
        else:
            reference_vectors.adapt(population.fitness)
    elif parameters.algorithm_name == "KRVEA":
        reference_vectors.adapt(population.fitness)
        problem.update(population)
