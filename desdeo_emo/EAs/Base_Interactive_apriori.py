import string
#from typing import Dict, Type, Union, Tuple

import numpy as np

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.EAs.BaseEA import BaseAprioriMOEA
from desdeo_emo.EAs.MOEAD import MOEA_D
from desdeo_problem import MOProblem
from desdeo_tools.scalarization.ASF import SimpleASF
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min



class eaError(Exception):
    """Raised when an error related to EA occurs"""


class BaseInteractiveApriori:
    """This class provides the basic structure for making an a priori algorithm interactive."""

    def __init__(
        self,
        problem: MOProblem,
        num_solutions_display: int,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
    ):
        """Initialize the method here."""
        self.problem: MOProblem = problem
        self.a_priori_method: BaseAprioriMOEA = None
        self.reference_point: np.array = None
        self.initialization_method: BaseDecompositionEA = MOEA_D
        self.interact: bool = True
        self.roi_size: float = 1
        self.num_solutions_display: int = num_solutions_display
        self.n_gen_per_iter: int = n_gen_per_iter
        self.total_function_evaluations = total_function_evaluations
        self.use_surrogates: bool = use_surrogates
        # Internal counters and state trackers
        self._iteration_counter: int = 0
        self._gen_count_in_curr_iteration: int = 0
        self._current_gen_count: int = 0
        self._function_evaluation_count: int = 0
        self._current_display_solutions: np.array = np.array([])
        self._ideal_point: np.array = np.array([])
        self._nadir_point: np.array = np.array([])
        #self.evolver: BaseDecompositionEA = None
        self._reference_point_list: np.array = np.empty((0,problem.n_of_objectives), float)
        self._shared_population: np.array = np.array([])

    def start(self):
        #get ideal and nadir points
        #initialize population
        evolver_initialization = self.initialization_method(self.problem, n_iterations=1, n_gen_per_iter=self.n_gen_per_iter)
        while evolver_initialization.continue_evolution():
            evolver_initialization.iterate()

        self._ideal_point = np.min(evolver_initialization.population.fitness, axis=0)
        self._nadir_point = np.max(evolver_initialization.population.fitness, axis=0)

        kmeans = KMeans(n_clusters=self.num_solutions_display, random_state=0).fit(evolver_initialization.population.objectives)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, evolver_initialization.population.objectives)
        self._current_display_solutions = evolver_initialization.population.objectives[closest]
        self._shared_population = evolver_initialization.population

    def end(self):
        """To be run at the end of the evolution process.
        """
        #select most relevant solutions
        #
        asf_values = SimpleASF([1]*self.problem.n_of_objectives).__call__(
        self.evolver.population.objectives, self.reference_point)
        idx = np.argpartition(asf_values, self.num_solutions_display)[:self.num_solutions_display] #indices of best solutions based on ASF
        self._current_display_solutions = self.evolver.population.objectives[idx] 

    def optimize(self):
        """Run the a priori EA
        """
        #reutilize best solutions
        #self.a_priori_method = self.a_priori_method()
        
        while self.a_priori_method.continue_evolution():
            self.a_priori_method.iterate()

    def cluster_ASF(self):
        asf_values = SimpleASF([1]*self.problem.n_of_objectives).__call__(
        self.evolver.population.objectives, self.reference_point)
        idx = np.argpartition(asf_values, self.num_solutions_display)[:self.num_solutions_display] #indices of best solutions based on ASF
        self._current_display_solutions = self.evolver.population.objectives[idx] 

    def set_preferences(self, new_reference_point, a_priori_evolver):
        self.current_reference_point = new_reference_point
        self._reference_point_list = np.append(self._reference_point_list, new_reference_point, axis=0)
        self.a_priori_method = a_priori_evolver



