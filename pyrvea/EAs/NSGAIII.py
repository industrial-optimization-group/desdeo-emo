from typing import TYPE_CHECKING

from pyrvea.Selection.NSGAIII_select import NSGAIII_select
from pyrvea.EAs.baseEA import BaseDecompositionEA
from pyrvea.OtherTools.ReferenceVectors import ReferenceVectors

import numpy as np

if TYPE_CHECKING:
    from pyrvea.Population.Population import Population


class NSGAIII(BaseDecompositionEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    [description]
    """

    def set_params(
        self,
        population: "Population" = None,
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = True,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 10,
        plotting: bool = True,
    ):
        lattice_resolution_options = {
            "2": 49,
            "3": 13,
            "4": 7,
            "5": 5,
            "6": 4,
            "7": 3,
            "8": 3,
            "9": 3,
            "10": 3,
        }
        if population.problem.num_of_objectives < 11:
            lattice_resolution = lattice_resolution_options[
                str(population.problem.num_of_objectives)
            ]
        else:
            lattice_resolution = 3
        reference_vectors = ReferenceVectors(
                lattice_resolution, population.problem.num_of_objectives
            )
        nsga3params = {
            "population_size": reference_vectors.number_of_vectors,
            "lattice_resolution": lattice_resolution,
            "interact": interact,
            "a_priori": a_priori_preference,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "ploton": plotting,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
            "reference_vectors": reference_vectors,
            "extreme_points": None,
        }
        return nsga3params

    def select(self, population: "Population"):
        Selection, extreme_points = NSGAIII_select(
            population.fitness,
            self.params["reference_vectors"].values_planar,
            population.ideal_fitness,
            population.worst_fitness,
            self.params["extreme_points"],
            self.params["population_size"]
        )
        self.params["extreme_points"] = extreme_points
        return Selection

    def _run_interruption(self, population: "Population"):
        """Run the interruption phase of RVEA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors, conducts interaction with the user.

        Parameters
        ----------
        population : Population
        """
        if self.params["interact"] or (
            self.params["a_priori"] and self.params["current_iteration_count"] == 1
        ):
            ideal = population.ideal_fitness
            refpoint = np.zeros_like(ideal)
            print("Ideal vector is ", ideal)
            for index in range(len(refpoint)):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    pref_val = float(
                        input("Please input a value worse than the ideal: ")
                    )
                    if pref_val > ideal[index]:
                        refpoint[index] = pref_val
                        break
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            self.params["reference_vectors"].iteractive_adapt_1(refpoint)
            self.params["reference_vectors"].add_edge_vectors()
        else:
            self.params["reference_vectors"].adapt(population.fitness)
        self.params["reference_vectors"].neighbouring_angles()

