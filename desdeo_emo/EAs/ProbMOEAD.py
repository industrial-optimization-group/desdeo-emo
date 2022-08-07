from typing import Dict

from numpy import argsort
from numpy import min as npmin
from numpy import vstack
from numpy import array
from numpy import where 
from scipy.spatial.distance import cdist as s_cdist

from numpy.random import permutation
from scipy.spatial import distance_matrix
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_problem.Problem import MOProblem

from desdeo_emo.selection import tournament_select

from desdeo_emo.selection.MOEAD_select import MOEAD_select

from desdeo_emo.selection.ProbMOEAD_select import ProbMOEAD_select
from desdeo_emo.selection.ProbMOEAD_select_v3 import ProbMOEAD_select_v3
from desdeo_emo.selection.ProbMOEAD_WS_select import ProbMOEAD_select as ProbMOEAD_select_WS
from desdeo_emo.selection.ProbMOEAD_TCH_select import ProbMOEAD_select as ProbMOEAD_select_TCH

from desdeo_emo.selection.HybMOEAD_select import HybMOEAD_select
from desdeo_emo.selection.HybMOEAD_select_v3 import HybMOEAD_select_v3

from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover
import copy


theta_min = 0
theta_max = 500

class MOEA_D(BaseDecompositionEA):
    """Python implementation of MOEA/D

    Parameters
    ----------
    problem: MOProblem
    	The problem class object specifying the details of the problem.
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
    SF_type: str, optional
    	One of ["TCH", "PBI", "WS"]. To be used as scalarizing function. TCH is used as default 
    	option.
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
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        self.problem = problem
        self.n_neighbors = n_neighbors
        self.SF_type = SF_type
        self.n_parents = n_parents
        #self.population.mutation = BP_mutation(problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds(), 0.5, 20)
        #self.population.recombination = SBX_xover(1.0, 20)
        self.population.mutation = BP_mutation(problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds())
        self.population.recombination = SBX_xover()
        selection_operator = MOEAD_select(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator
        # Compute the distance between each pair of reference vectors
        #print (len(self.reference_vectors.values))
        distance_matrix_vectors = distance_matrix(self.reference_vectors.values, self.reference_vectors.values)
        # Get the closest vectors to obtain the neighborhoods
        self.neighborhoods = argsort(distance_matrix_vectors, axis=1, kind='quicksort')[:,:n_neighbors]
        self.population.update_ideal()
        self._ideal_point = self.population.ideal_objective_vector

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            # set adaptive theta
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
            
        #TODO: check this----------------
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1
    
    def _select(self, current_neighborhood, offspring_fx, theta_adaptive) -> list:
        return self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, theta_adaptive)

class ProbMOEAD(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = ProbMOEAD_select(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            # set adaptive theta
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

class HybMOEAD(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = HybMOEAD_select(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

class ProbMOEAD_v3(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = ProbMOEAD_select_v3(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

class ProbMOEAD_WS(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = ProbMOEAD_select_WS(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

class ProbMOEAD_TCH(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = ProbMOEAD_select_TCH(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

class HybMOEAD_v3(MOEA_D):
    def __init__(  #parameters of the class
        self,
        problem: MOProblem,
        population_size: int = None,
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        SF_type: str = "PBI",
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
        super().__init__( #parameters for decomposition based approach
            problem = problem,
            population_size = population_size,
            population_params = population_params,
            initial_population = initial_population,
            lattice_resolution = lattice_resolution,
            a_priori = a_priori,
            interact = interact,
            use_surrogates = use_surrogates,
            n_iterations = n_iterations,
            n_gen_per_iter = n_gen_per_iter,
            total_function_evaluations = total_function_evaluations,
        )
        self.population_size = self.population.pop_size
        #self.population_size = population_size
        selection_operator = HybMOEAD_select_v3(
            self.population, SF_type=SF_type
        )
        self.selection_operator = selection_operator

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i,:]
            selected_parents     = current_neighborhood[permutation(self.n_neighbors)][:self.n_parents]


            offspring = self.population.recombination.do(self.population.individuals, selected_parents)
            offspring = self.population.mutation.do(offspring)
            # Apply genetic operators over two random individuals
            #offspring = self.population.mate(selected_parents)
            offspring = array(offspring[0,:])
            
            # Repair the solution if it is needed
            offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off     =  self.problem.evaluate(offspring, self.use_surrogates)
            offspring_fx    =  results_off.objectives
            offspring_unc   =  results_off.uncertainity
            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = npmin(vstack([self._ideal_point, offspring_fx]), axis=0)
            theta_adaptive = theta_min + (theta_max - theta_min) * (self._function_evaluation_count / self.total_function_evaluations)
            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        indx = copy.deepcopy(self.population.individuals)
        objx = copy.deepcopy(self.population.objectives)
        uncx = copy.deepcopy(self.population.uncertainity)
        self.population.individuals_archive[str(self.population.gen_count)] = indx
        self.population.objectives_archive[str(self.population.gen_count)] = objx
        self.population.uncertainty_archive[str(self.population.gen_count)] = uncx
        self.population.gen_count += 1

    def _select(self, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> list:
        zz= self.selection_operator.do(self.population,self.reference_vectors,self._ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive)
        #print("Selection:",zz)
        return zz

