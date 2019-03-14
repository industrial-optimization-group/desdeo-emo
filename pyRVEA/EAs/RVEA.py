from typing import TYPE_CHECKING

from pyRVEA.Selection.APD_select import APD_select
from pyRVEA.EAs.baseEA import BaseDecompositionEA
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors

import numpy as np

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population


class RVEA(BaseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    See the details of RVEA in the following paper

    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

    The source code of pyRVEA is implemented by Bhupinder Saini

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi

    Project researcher at University of Jyväskylä.
    """

    def set_params(
        self,
        population: "Population" = None,
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = False,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 10,
        Alpha: float = 2,
        plotting: bool = True,
    ):
        """Set up the parameters. Save in RVEA.params. Note, this should be
        changed to align with the current structure.

        Parameters
        ----------
        population : Population
            Population object
        population_size : int
            Population Size
        lattice_resolution : int
            Lattice resolution
        interact : bool
            bool to enable or disable interaction. Enabled if True
        a_priori_preference : bool
            similar to interact
        generations_per_iteration : int
            Number of generations per iteration.
        iterations : int
            Total Number of iterations.
        Alpha : float
            The alpha parameter of APD selection.
        plotting : bool
            Useless really.
        Returns
        -------

        """
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
            "reference_vectors": ReferenceVectors(
                lattice_resolution, population.problem.num_of_objectives
            ),
        }
        return rveaparams

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
            self.params["reference_vectors"].iteractive_adapt_1(refpoint)
            self.params["reference_vectors"].add_edge_vectors()
        else:
            self.params["reference_vectors"].adapt(population.fitness)
        self.params["reference_vectors"].neighbouring_angles()

    def select(self, population: "Population"):
        """Describe a selection mechanism. Return indices of selected
        individuals.

        # APD Based selection. # This is different from the paper. #
        params.genetations != total number of generations. This is a compromise.
        Also this APD uses an archived ideal point, rather than current, potentially
        worse ideal point.

        Parameters
        ----------
        population : Population
            Population information

        Returns
        -------
        list
            list: Indices of selected individuals.
        """
        penalty_factor = (
            (self.params["current_iteration_gen_count"] / self.params["generations"])
            ** self.params["Alpha"]
        ) * population.problem.num_of_objectives
        return APD_select(
            fitness=population.fitness,
            vectors=self.params["reference_vectors"],
            penalty_factor=penalty_factor,
            ideal=population.ideal_fitness,
        )
