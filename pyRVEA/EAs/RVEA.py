from typing import TYPE_CHECKING

from pyRVEA.Selection.APD_select import APD_select
from pyRVEA.EAs.baseEA import baseDecompositionEA
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors

import numpy as np

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population


class RVEA(baseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    See the details of RVEA in the following paper

    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
    A Reference Vector Guided Evolutionary Algorithm for Many-objective
    Optimization, IEEE Transactions on Evolutionary Computation, 2016

    The source code of pyRVEA is implemented by Bhupinder Saini

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi
    Project researcher at University of Jyväskylä.
    """

    def __init__(self, population: "Population"):
        """
        Initialize RVEA.

        This will set up the parameters of RVEA, create Reference Vectors, and
        (as of Feb 2019) run the first iteration of RVEA.

        Parameters
        ----------
        population : Population
            This variable is updated as evolution takes place, also contains problem.

        Returns
        -------
        Population
            Returns the Population after evolution.

        """
        self.params = self.set_params(
            num_of_objectives=population.problem.num_of_objectives
        )
        self.reference_vectors = self.create_reference_vectors(
            population.problem.num_of_objectives
        )
        self._next_iteration(population)

    def set_params(
        self,
        population_size: int = None,
        lattice_resolution: int = None,
        num_of_objectives: int = None,
        interact: bool = False,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 10,
        Alpha: float = 2,
        plotting: bool = True,
        algorithm_name="RVEA",
    ):
        """Set up the parameters. Save in RVEA.params"""
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
        if num_of_objectives < 11:
            lattice_resolution = lattice_resolution_options[str(num_of_objectives)]
        else:
            lattice_resolution = 3
        rveaparams = {
            "population_size": population_size,
            "lattice_resolution": lattice_resolution,
            "interact": interact,
            "a_priori": a_priori_preference,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "Alpha": Alpha,
            "ploton": plotting,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
        }
        return rveaparams

    def create_reference_vectors(self, num_of_objectives):
        """Create reference vectors."""
        return ReferenceVectors(self.params["lattice_resolution"], num_of_objectives)

    def _next_iteration(self, population: "Population"):
        """Run one iteration of RVEA.

        One iteration consists of a constant or variable number of generations. This
        method leaves RVEA.params unchanged.

        Parameters
        ----------
        population : Population

        """
        self.params["current_iteration_gen_count"] = 0
        while self.continuegenerations():
            self.reference_vectors.neighbouring_angles()
            self._next_gen_(population)
            self.params["current_iteration_gen_count"] += 1
        self.params["current_iteration_count"] += 1

    def _next_gen_(self, population: "Population"):
        """Run one generation of RVEA.

        This method leaves method.params unchanged.
        Intended to be used by next_iteration.

        Parameters
        ----------
        population : Population

        """
        offspring = population.mate()
        population.add(offspring)
        # APD Based selection.
        # This is different from the paper.
        # params.genetations != total number of generations. This is a compromise.
        penalty_factor = (
            (self.params["current_iteration_gen_count"] / self.params["generations"])
            ** self.params["Alpha"]
        ) * population.problem.num_of_objectives
        selected = self.select(
            population.fitness, self.reference_vectors, penalty_factor
        )
        population.keep(selected)
        # return population?

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
            # refpoint = np.mean(population.fitness, axis=0)
            ideal = population.ideal
            nadir = population.nadir
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
            self.reference_vectors.iteractive_adapt_1(refpoint)
            self.reference_vectors.add_edge_vectors()
        else:
            self.reference_vectors.adapt(population.fitness)

    def select(self, *args):
        """Describe a selection mechanism. Return indices of selected individuals."""
        return APD_select(*args)

    def continuegenerations(self):
        return self.params["current_iteration_gen_count"] < self.params["generations"]
