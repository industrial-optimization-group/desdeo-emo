from typing import TYPE_CHECKING

from pyrvea.Selection.NSGAIII_select import NSGAIII_select
from pyrvea.EAs.baseEA import BaseDecompositionEA
from pyrvea.OtherTools.ReferenceVectors import ReferenceVectors


if TYPE_CHECKING:
    from pyrvea.Population.Population import Population


class NSGAIII(BaseDecompositionEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    [description]
    """

    def set_params(
        self,
        population: "Population",
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = False,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 10,
        iterations: int = 10,
        plotting: bool = True,
        logging: bool = False,
        logfile=None,
        **kwargs
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
            "logging": logging,
            "logfile": logfile,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
            "current_total_gen_count": 0,
            "total_generations": iterations * generations_per_iteration,
            "reference_vectors": reference_vectors,
            "extreme_points": None,
        }
        nsga3params.update(kwargs)
        return nsga3params

    def select(self, population: "Population"):
        Selection, extreme_points = NSGAIII_select(
            population.fitness,
            self.params["reference_vectors"].values_planar,
            population.ideal_fitness,
            population.worst_fitness,
            self.params["extreme_points"],
            self.params["population_size"],
        )
        self.params["extreme_points"] = extreme_points
        return Selection
