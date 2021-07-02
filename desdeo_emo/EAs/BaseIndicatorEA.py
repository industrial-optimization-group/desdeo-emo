from typing import Dict, Type, Union, Tuple
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_problem.Problem import MOProblem

from desdeo_problem.Problem import DataProblem

import matplotlib.pyplot as plt
from desdeo_problem.Problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs import BaseEA
from numba import njit 
import hvwfg as hv
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_emo.selection.tournament_select import tour_select

import plotly.graph_objects as go
from pyDOE import lhs
from pygmo import non_dominated_front_2d as nd2

@njit 
def epsilon_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.
    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to. 
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be set of arrays.
    Returns: 
        float: The factor by which the approximating front is worse than the reference front with respect to all 
        objectives.
    """
    eps = 0.0
    ref_len = len(reference_front)
    front_len = len(front)
    # number of objectives
    num_obj = len(front[0])

    for i in range(ref_len):
        for j in range(front_len):
            for k in range(num_obj):
                value = front[j][k] - reference_front[i][k]
                if value > eps:
                    eps = value

    return eps

def hypervolume_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the hypervolume-indicator between reference front and current approximating point.
    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to. 
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be 2D array.
    Returns: 
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    return hv.wfg(reference_front, front.reshape(-1))


# this is probably slower than needed
def binary_tournament_select(population:Population) -> list:
        parents = []
        for i in range(int(population.pop_size / 2)):
            parents.append(
                np.asarray(
                    tour_select(population.fitness[:, 0], 2),
                    tour_select(population.fitness[:, 0], 2),
            ))
        return parents



# this is a bigger problem to get Base class to work. but this is a some sort of a start
class BaseIndicatorEA(BaseEA):
    """

    """
    # let's start commenting and removing what we do now know if we need
    # assuming desdeos BP-mutation is correct to use here.
    def __init__(
        self,
        problem: MOProblem,
        selection_operator: Type[SelectionBase] = None,
        population_size: int = None, # size required
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        indicator: int = 0, # 0 for additive eps and 1 for hypervolume-indicator
    ):
        super().__init__(
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            selection_operator=selection_operator,
            use_surrogates=use_surrogates,
        )

        if initial_population is not None:
            self.population = initial_population
        elif initial_population is None:
            if population_size is None:
                population_size = 100 # keksi parempi
            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size
        
        #print("Using BaseIndicatorEA init")
        
    # täälä operaattorit in the main loop of the algoritmh
    def _next_gen(self):
        # call _fitness_assigment (using indicator)
        self._fitness_assignment()
        # iterate until size of population less than alpha. Dunno how to implement this. maybe stopid while loop or smh?
        # call the _select fucntion, delete the worst individual
        while (self.population.pop_size <= self.population.individuals.shape[0]):
            selected = self._select()
            self.population.delete(selected)
            # update the fitness values of remaining individuals
            self._fitness_assignment()
        # check termination
        # currently just runs until func evaluations run out.

        # perform binary tournament selection. in these steps 5 and 6 we give offspring to the population and make it bigger. kovakoodataan tämä nytten, mietitään myöhemmin sitten muuten.
        chosen = binary_tournament_select(self.population)        

        # variation, call the recombination operators
        offspring = self.population.mate(mating_individuals=chosen)
        self.population.add(offspring)

        self._current_gen_count += 1
        #print(self._current_gen_count)
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]
        print(self._function_evaluation_count)


    # need to implement the enviromental selection. Only calls it from selection module
    def _select(self) -> list:
        return self.selection_operator.do(self.population)

    #implements fitness computing. 
    # TODO: trouble of calling different indicators with the design since indicators are just functions. Let's cross that bridge when we come to it.
    # no idea if this works correctly
    # TODO: what about calculating the fitness other objective? does it work? It looks to only calculate for the .fitness[0]
    def _fitness_assignment(self):
        population = self.population
        pop_size = population.individuals.shape[0]
        for i in range(pop_size):
            population.fitness[i] = 0,0 # dunno if needed
            for j in range(pop_size):
                if j != i:
                    population.fitness[i] += -np.exp(-epsilon_indicator([population.objectives[i]], [population.objectives[j]]) / 0.05)



# kappa is a problem, how to use it in BaseIndicatorEA
class IBEA(BaseIndicatorEA):
    def __init__(self,
        problem: MOProblem,
        population_size: int = None, # size required
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        # what ibea needs
        kappa: float = 0.05, # fitness scaling ratio
        indicator: int = 0,
                 ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            initial_population=initial_population,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
            indicator = indicator,
        )
        
        self.kappa = kappa
        self.indicator = indicator
        selection_operator = EnvironmentalSelection(self.population)
        self.selection_operator = selection_operator

        #print("using IBEA")

    
# population.fitness on 100,2 koska kaks objektivea, eli jokaisella kaksi eri arvoa.
if __name__=="__main__":
    # start with simpler example you can calculate by hand to confirm it works.
    # get the problem
    problem_name = "ZDT1" # needs 30,100
    #problem_name = "ZDT2" 
    #problem_name = "DTLZ2" 
    #problem_name = "DTLZ7" 

    problem = test_problem_builder(problem_name)

    evolver = IBEA(problem, n_iterations=10,n_gen_per_iter=100, total_function_evaluations=25000)
    
    print("starting front", evolver.population.objectives[0::10])
    while evolver.continue_evolution():
        evolver.iterate()

    front_true = evolver.population.objectives
    print(front_true[0::10])

    true = plt.scatter(x=front_true[:,0], y=front_true[:,1], label="True Front")
    plt.title(f"Fronts obtained with various algorithms on the problem")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
