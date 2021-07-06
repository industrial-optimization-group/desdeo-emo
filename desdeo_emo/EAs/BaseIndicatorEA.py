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


# tämän pitäisi olla nopea
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
        for i in range(int(population.pop_size)): # maybe half this or quarter?
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
        # call _fitness_assigment (using indicator). replacement
        self._fitness_assignment()
        # iterate until size of population less than alpha. Dunno how to implement this. maybe stopid while loop or smh?
        # call the _select fucntion, delete the worst individual
        while (self.population.pop_size < self.population.individuals.shape[0]):

            # maybe refactor the join_pop, to make simpler and possibly find speedgain

            # choose individual with smallest fitness value
            selected = self._select()
            worst_index = selected[0]

            # update the fitness values
            poplen = self.population.individuals.shape[0]
            for i in range(poplen):
                self.population.fitness[i] += np.exp(-epsilon_indicator([self.population.objectives[i]], [self.population.objectives[worst_index]]) / 0.05)

            # remove the worst individula 
            self.population.delete(selected)

            # update the fitness values of remaining individuals. Here we should use worst index.. also would probably make it faster.
           # self._fitness_assignment()
        # check termination
        # currently just runs until func evaluations run out.

        # perform binary tournament selection. in these steps 5 and 6 we give offspring to the population and make it bigger. kovakoodataan tämä nytten, mietitään myöhemmin sitten muuten.
        # this might not be correct either
        chosen = binary_tournament_select(self.population)        

        # variation, call the recombination operators
        offspring = self.population.mate(mating_individuals=chosen)
        self.population.add(offspring)

        self._current_gen_count += 1
        #print(self._current_gen_count)
        self._gen_count_in_curr_iteration += 1
        #print(f"gen count curr ite {self._gen_count_in_curr_iteration}")
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
        #pop_width = population.fitness.shape[1]

        print(f"pop fitness ka: {np.mean(population.fitness)}, pop fitness kh: {np.std(population.fitness)}")

        fit_comp = np.zeros([pop_size, pop_size])
        maxIndicatorVal = 0
        # TODO: population.fitness[i] could be the issue, since it has objective number of values but we put the indicator value to all of them.
        for i in range(pop_size):
            #population.fitness[i] = [0]*pop_width # 0 all the fitness values. 
            for j in range(pop_size):
                if j != i:
                    population.fitness[i] += -np.exp(-epsilon_indicator([population.objectives[i]], [population.objectives[j]]) / 0.05)

# Ibea.c :stä. ei onnistu tämäkään
#                fit_comp[i][j] = epsilon_indicator([population.objectives[i]], [population.objectives[j]])
#                fitabs =  np.abs(fit_comp[i][j])
#                if maxIndicatorVal < fitabs:
#                        maxIndicatorVal = fitabs
#                
#        for i in range(pop_size):
#            for j in range(pop_size):
#                fit_comp[i][j] = np.exp((-fit_comp[i][j]/maxIndicatorVal)/0.05)
#                #print(fit_comp[i][j])
#
#        for i in range(pop_size):
#            summa = 0
#            for j in range(pop_size):
#                if i != j:
#                    summa += fit_comp[i][j]
#                    population.fitness[i] = summa
#

                    #print(population.fitness[i][k])



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

    

"""
10 000 evals
ZDT1 works
ZDT2 almost works 
ZDT3 works
ZDT4 doesn't work
ZDT6 doestn really work- kinda works with more evals
"""
def testZDTs():
    problem_name = "ZDT1" # needs 30,100. ZDT1 seems to converge even with about 2000 total_function_evaluations
    #problem_name = "ZDT3" # seems work ok.
    #problem_name = "ZDT6" # this just starts going worse and worse 
    # doesn't work properly with ZDT4... atleast saves too many bad solutions..

    problem = test_problem_builder(problem_name)
    evolver = IBEA(problem, n_iterations=10,n_gen_per_iter=100, total_function_evaluations=10000)
    
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


# TODO: starting to feel this IBEA has some problems still.. shoudl be doing better with less evals
# TODO: I broke it at some point. No idea whattt is wrong
# works for these too
"""
15 000 evals
DTLZ1 doesn't work
DTLZ2 kinda works
DTLZ6 doesnt work
DTLZ7 kinda works. 
"""
def testDTLZs():
    #problem_name = "DTLZ1" 
    #problem = test_problem_builder(problem_name, n_of_variables=7, n_of_objectives=3)
        
    #problem_name = "DTLZ2" # seems to work okay?, even with low total_function_evaluations. po sols are not that even in places tho.
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    #problem_name = "DTLZ6" # does not do that good.. mean and std get low but then they start oscillating
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    problem_name = "DTLZ7" # this looks pretty good, same as dtlz6 for the mean and std 
    problem = test_problem_builder(problem_name, n_of_variables=22, n_of_objectives=3)

    evolver = IBEA(problem, n_iterations=10, n_gen_per_iter=100, total_function_evaluations=15000)
    
    print("starting front", evolver.population.objectives[0::10])
    while evolver.continue_evolution():
        evolver.iterate()

    front_true = evolver.population.objectives
    print(front_true[0::10])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30,45)
    ax.scatter(front_true[:,0],front_true[:,1],front_true[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



# population.fitness on 100,2 koska kaks objektivea, eli jokaisella kaksi eri arvoa.
if __name__=="__main__":
   testZDTs()
   #testDTLZs()
