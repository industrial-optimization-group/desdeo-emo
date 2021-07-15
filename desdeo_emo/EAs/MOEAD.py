from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist as s_cdist

from numpy.random import permutation
from scipy.spatial import distance_matrix
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_problem.Problem import MOProblem

from desdeo_emo.selection import tournament_select
from desdeo_emo.selection.MOEAD_select import MOEAD_select
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover

from desdeo_tools.scalarization import MOEADSF
from desdeo_tools.scalarization.MOEADSF import Tchebycheff, PBI


class MOEA_D(BaseDecompositionEA):
    """Python implementation of MOEA/D

    .. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," 
    in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.

    Parameters
    ----------
    problem: MOProblem
    	The problem class object specifying the details of the problem.
    scalarization_function: MOEADSF
    	The scalarization function to compare the solutions. Some implementations 
        can be found in desdeo-tools/scalarization/MOEADSF. By default it uses the
        PBI function.
    n_neighbors: int, optional
    	Number of reference vectors considered in the neighborhoods creation. The default 
    	number is 20.
    population_params: Dict, optional
    	The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population: Population, optional
    	An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    lattice_resolution: int, optional
    	The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    n_parents: int, optional
    	Number of individuals considered for the generation of offspring solutions. The default
    	option is 2.
    a_priori: bool, optional
    	A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact: bool, optional
    	A bool variable defining whether interactive preference is to be used or
        not. By default False
    use_surrogates: bool, optional
    	A bool variable defining whether surrogate problems are to be used or
        not. By default False
    n_iterations: int, optional
     	The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter: int, optional
    	The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations: int, optional
    	Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(  # parameters of the class
        self,
        problem: MOProblem,
        scalarization_function: MOEADSF = PBI(),
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        use_repair: bool = True,
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__(  # parameters for decomposition based approach
            problem=problem,
            population_size=None,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
        )
        self.population_size = self.reference_vectors.number_of_vectors
        self.problem = problem
        self.scalarization_function = scalarization_function
        self.n_neighbors = n_neighbors

        self.use_repair = use_repair
        self.n_parents = n_parents

        selection_operator = MOEAD_select(
            self.population, SF_type=self.scalarization_function
        )
        self.selection_operator = selection_operator
        # Compute the distance between each pair of reference vectors
        distance_matrix_vectors = distance_matrix(
            self.reference_vectors.values_planar, self.reference_vectors.values_planar
        )
        # Get the closest vectors to obtain the neighborhoods
        self.neighborhoods = np.argsort(
            distance_matrix_vectors, axis=1, kind="quicksort"
        )[:, :n_neighbors]
        self.population.update_ideal()
        self._ideal_point = self.population.ideal_fitness_val

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i, :]
            selected_parents = current_neighborhood[permutation(self.n_neighbors)][
                : self.n_parents
            ]

            # Apply genetic operators over two random individuals
            offspring = self.population.mate(selected_parents)
            offspring = offspring[0, :]

            # Repair the solution if it is needed
            if self.use_repair:
                offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off = self.problem.evaluate(offspring, self.use_surrogates)

            offspring_fx = results_off.fitness[0, :]

            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = np.min(
                np.vstack([self._ideal_point, offspring_fx]), axis=0
            )

            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx)

            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1

    def _select(self, current_neighborhood, offspring_fx) -> list:
        return self.selection_operator.do(
            self.population,
            self.reference_vectors,
            self._ideal_point,
            current_neighborhood,
            offspring_fx,
        )

