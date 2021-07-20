from typing import Dict, Type, Union, Tuple, Callable
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase

import matplotlib.pyplot as plt
from desdeo_problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs import BaseEA
from numba import njit 
import hvwfg as hv
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_emo.selection.tournament_select import tour_select
from desdeo_tools.scalarization import SimpleASF, ReferencePointASF

# TODO: remember to sort import

from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_emo.EAs.IBEA import IBEA

from desdeo_tools.utilities.quality_indicator import epsilon_indicator, epsilon_indicator_ndims



# find proper place, would be neat if it could easily be put in place of eps indi.. to desdeo_tools again we go i think
# Ip(y, x) =) Ie(y,x) / s(g, f(x), delta)
def preference_indicator(ref_front:np.ndarray, front:np.ndarray, ref_point:np.ndarray, delta: np.float64):
    # normalize still, 0.01 is delta now
    ref_front = np.array(ref_front, dtype=np.float64)
    front = np.array(front, dtype=np.float64)
    xasf = SimpleASF(front)
    yasf = SimpleASF(ref_front)

    norm = xasf(front, reference_point=ref_point) + delta - np.min(yasf(ref_front, reference_point=ref_point))
    #eps = epsilon_indicator(ref_front, front)
    #res = eps/norm
    #return res
    return epsilon_indicator(ref_front, front)/norm




# for start we can implement stuff here too
class PBEA(BaseIndicatorEA):
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
        # what pbea needs
        kappa: np.float64 = 0.05, # fitness scaling ratio
        indicator: Callable = preference_indicator, # default indicator is epsilon_indicator
        reference_point = None,
        delta: float = 0.1, # spesifity
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
            indicator=indicator,
            reference_point=reference_point,
        )
        
        self.kappa = kappa
        self.delta = delta
        self.indicator = indicator # needs preference indicator
        self.reference_point = reference_point
        print(self.reference_point)
        selection_operator = EnvironmentalSelection(self.population)
        self.selection_operator = selection_operator

        print("using PBEA")

    # run ibea in start ?
    def start(self):
        pass
        print(self.population.fitness)
        



def inter_zdt():
    problem_name = "ZDT3" # seems work ok.
    #problem_name = "ZDT6" # this just starts going worse and worse 

    problem = test_problem_builder(problem_name)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=35, n_iterations=3, n_gen_per_iter=100,total_function_evaluations=1000)
    while ib.continue_evolution():
        ib.iterate()
    individuals, objective_values = ib.end()

    # need to get the population
    print(ib.return_pop())
    ini_pop = ib.return_pop()

    # step 1: reference point. TODO: actually ask from DM
    #ref_point = np.array([0.9,-0.2], dtype=np.float64) # we want the solutions at middle 
    delta = 0.1
    # step 2: local approximation
    evolver = PBEA(problem, interact=True, population_size=35, initial_population=ini_pop, 
                   n_iterations=5, n_gen_per_iter=100, total_function_evaluations=1000, 
                   indicator=preference_indicator, delta=delta)
    
    #while evolver.continue_evolution():
        #evolver.iterate()
    pref, plot = evolver.requests()
    #print(pref, plot)
    print(pref.content['message'])
    # desdeo's logic doesnt make yet sense so this won't work
    pref.response = np.array([0.9,-0.2], dtype=np.float64) # we want the solutions at middle 

    pref, plot = evolver.iterate(pref)


"""
8 000 evals
"""
def testZDTs():
    #problem_name = "ZDT1" # needs 30,100. ZDT1 seems to converge even with about 2000 total_function_evaluations
    problem_name = "ZDT3" # seems work ok.
    #problem_name = "ZDT6" # this just starts going worse and worse 

    problem = test_problem_builder(problem_name)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=35, n_iterations=3, n_gen_per_iter=100,total_function_evaluations=3000)
    while ib.continue_evolution():
        ib.iterate()
    individuals, objective_values = ib.end()

    # need to get the population
    print(ib.return_pop())
    ini_pop = ib.return_pop()

    # should select small set of solutions to show to DM. For now we show all.
    true = plt.scatter(x=objective_values[:,0], y=objective_values[:,1], label="IBEA Front")
    plt.title(f"Fronts")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    #plt.show()

    # step 1: reference point. TODO: actually ask from DM
    ref_point = np.array([0.9,-0.2], dtype=np.float64) # we want the solutions at middle 
    delta = 0.1
    # step 2: local approximation
    evolver = PBEA(problem, population_size=35, initial_population=ini_pop, n_iterations=5, n_gen_per_iter=100, total_function_evaluations=1000, indicator=preference_indicator, reference_point=ref_point, delta=delta)
    while evolver.continue_evolution():
        evolver.iterate()

    # step 3: show result to the DM
    individuals2, objective_values2 = evolver.end()    
    print(ib.return_pop())
    ini_pop = evolver.return_pop()

    # should select small set of solutions to show to DM. For now we show all.
    true2 = plt.scatter(x=objective_values2[:,0], y=objective_values2[:,1], label="PBEA Front, ite1")
    plt.title(f"Fronts")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    #plt.show()


    # second iteration
    # step 1: reference point. TODO: actually ask from DM
    ref_point = np.array([0.9,-0.9], dtype=np.float64) # we want the solutions at middle 
    delta = 0.01
    # step 2: local approximation
    evolver = PBEA(problem, population_size=35, initial_population=ini_pop, n_iterations=5, n_gen_per_iter=100, total_function_evaluations=2000, indicator=preference_indicator, reference_point=ref_point, delta=delta)
    while evolver.continue_evolution():
        evolver.iterate()

    # step 3: show result to the DM
    individuals3, objective_values3 = evolver.end()
    print(ib.return_pop())
    ini_pop = evolver.return_pop()
    # should select small set of solutions to show to DM. For now we show all.
    true2 = plt.scatter(x=objective_values3[:,0], y=objective_values3[:,1], label="PBEA Front, ite2")
    plt.title(f"Fronts")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()

# dunno how pbea works with dtlzs
def testDTLZs():
    # sometimes the index error too
    #problem_name = "DTLZ1" 
    #problem = test_problem_builder(problem_name, n_of_variables=7, n_of_objectives=3)
        
    # with PBEA, in worst_index part selection fails to find anything, most likely due to earlier overflows
    #problem_name = "DTLZ2" # seems to work okay?, even with low total_function_evaluations. po sols are not that even in places tho.
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    #problem_name = "DTLZ4" # seems to work okay?, even with low total_function_evaluations. po sols are not that even in places tho.
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
    
    problem_name = "DTLZ6" # does not do that good.. mean and std get low but then they start oscillating
    problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    #problem_name = "DTLZ7" # this looks pretty good, same as dtlz6 for the mean and std 
    #problem = test_problem_builder(problem_name, n_of_variables=22, n_of_objectives=3)

    ib = IBEA(problem, population_size=100, n_iterations=10, n_gen_per_iter=100, total_function_evaluations=15000)
    
    #print("starting front", evolver.population.objectives[0::10])
    while ib.continue_evolution():
        ib.iterate()
        
    individuals, objective_values= ib.end()
    ini_pop = ib.return_pop()
    #print(result)
    #front_true = result[1]
    #print(front_true[0::10])
    # should select small set of solutions to show to DM. For now we show all.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30,45)
    ax.scatter(objective_values[:,0],objective_values[:,1],objective_values[:,2], label="IBEA front")
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    plt.legend()
    #plt.show()


    # step 1: reference point. TODO: actually ask from DM
    ref_point = np.array([0.6,0.7, 0.8], dtype=np.float64) # we want the solutions at middle 
    delta = 0.1
    # step 2: local approximation
    evolver = PBEA(problem, population_size=100, initial_population=ini_pop, n_iterations=5, n_gen_per_iter=100, total_function_evaluations=5000, indicator=preference_indicator, reference_point=ref_point, delta=delta)
    while evolver.continue_evolution():
        evolver.iterate()

    # step 3: show result to the DM
    individuals2, objective_values2 = evolver.end()    
    ini_pop = evolver.return_pop()

    # should select small set of solutions to show to DM. For now we show all.
    ax.scatter(objective_values2[:,0],objective_values2[:,1],objective_values2[:,2], label="PBEA front 1")
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    plt.legend()
    #plt.show()

    # second iteration
    # step 1: reference point. TODO: actually ask from DM
    ref_point = np.array([0.5,0.5,0.5], dtype=np.float64) # we want the solutions at middle 
    delta = 0.01
    # step 2: local approximation
    evolver = PBEA(problem, population_size=100, initial_population=ini_pop, n_iterations=5, n_gen_per_iter=100, total_function_evaluations=2000, indicator=preference_indicator, reference_point=ref_point, delta=delta)
    while evolver.continue_evolution():
        evolver.iterate()

    # step 3: show result to the DM
    individuals3, objective_values3 = evolver.end()
    ini_pop = evolver.return_pop()
    # should select small set of solutions to show to DM. For now we show all.
    ax.scatter(objective_values3[:,0],objective_values3[:,1],objective_values3[:,2], label="PBEA front 2")
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    plt.legend()
    plt.show()


# TODO: 
# domination comparison for fitness/objective vectors
if __name__=="__main__":

    #inter_zdt()
    testZDTs()
    #testDTLZs()

   #import cProfile
   #cProfile.run('testDTLZs()', "output.dat")

   #import pstats
   #from pstats import SortKey

   #with open('output_time.txt', "w") as f:
   #    p = pstats.Stats('output.dat', stream=f)
   #    p.sort_stats('time').print_stats()

   #with open('output_calls.txt', "w") as f:
   #    p = pstats.Stats('output.dat', stream=f)
   #    p.sort_stats('calls').print_stats()
