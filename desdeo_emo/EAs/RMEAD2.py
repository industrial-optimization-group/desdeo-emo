from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist as s_cdist

from numpy.random import permutation
from scipy.spatial import distance_matrix
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_problem import MOProblem

from desdeo_emo.selection.RMOEAD_select import RMEAD2_select
from desdeo_emo.selection.TournamentSelection import TournamentSelection
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover

from desdeo_tools.scalarization import MOEADSF
from desdeo_tools.scalarization.MOEADSF import Tchebycheff, PBI

class RMEAD2(BaseDecompositionEA):
    """Python implementation of RMOEA/D

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
        reference_point: np.array,
        roi_size: float,
        scalarization_function: MOEADSF = PBI(),
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
        self.reference_point = reference_point
        self.roi_size = roi_size
        self.problem = problem
        self.scalarization_function = scalarization_function

        self.parent_selection = TournamentSelection
        self.crossover_method = SBX_xover
        self.mutation_method = BP_mutation
        self.use_repair = use_repair
        self.n_parents = n_parents

        selection_operator = RMEAD2_select(
            pop=self.population, ref_point=self.reference_point
        )
        self.selection_operator = selection_operator
        self.population.update_ideal()
        self._ideal_point = self.population.ideal_fitness_val

    def _next_gen(self):

        SF_values = self.scalarization_function(self.population.fitness, self.reference_vectors.values_planar, self.reference_point)

        parents = self.parent_selection.do(self.population, SF_values)
        offspring = self.crossover_method.do(self.population, parents)
        offspring = self.mutation_method.do(offspring)

        
        offspring = self.population.mate()  # (params=self.params)
        self.population.add(offspring, self.use_surrogates)


        #measure Euclidean distance between the RP and the population

        #get the closest solution

        #get the reference vector asigned to the closest solution


        #update reference vectors

        selected = self._select()
        self.population.keep(selected)

        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]



    def _select(self, current_neighborhood, offspring_fx) -> list:
        return self.selection_operator.do(
            self.population,
            self.reference_vectors,
            current_neighborhood,
            offspring_fx,
        )

    def adapt_reference_vectors(self):
        ideal = self.population.ideal_fitness_val
        refpoint = (self.reference_point * self.population.problem._max_multiplier)
        refpoint = refpoint - ideal
        norm = np.sqrt(np.sum(np.square(refpoint)))
        refpoint = refpoint / norm
        self.reference_vectors.iteractive_adapt_3(refpoint)
        return 0

