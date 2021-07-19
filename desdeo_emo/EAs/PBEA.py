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



# find proper place, would be neat if it could easily be put in place of eps indi
def preference_indicator(y:np.ndarray, x:np.ndarray):
    # normalize still, 0.01 is delta now
    ref_point = np.array([0.5,0.5]) # we want the solutions at middle 
    xasf = SimpleASF(x)

    yasf = SimpleASF(y)
    norm = xasf(x, reference_point=ref_point) + 0.1 - np.min(yasf(y, reference_point=ref_point))
    # Ip(y, x) =) Ie(y,x) / s(g, f(x), delta)
    return epsilon_indicator(y, x)/ norm




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
        # what ibea needs
        kappa: float = 0.05, # fitness scaling ratio
        indicator: Callable = preference_indicator, # default indicator is epsilon_indicator
        reference_point = None,
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
        



"""
8 000 evals
"""
def testZDTs():
    #problem_name = "ZDT1" # needs 30,100. ZDT1 seems to converge even with about 2000 total_function_evaluations
    #problem_name = "ZDT3" # seems work ok.
    problem_name = "ZDT6" # this just starts going worse and worse 
    # doesn't work properly with ZDT4... atleast saves too many bad solutions..

    problem = test_problem_builder(problem_name)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=40, total_function_evaluations=5000)
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
    ref_point = np.array([0.5,0.5]) # we want the solutions at middle 

    # step 2: local approximation
    evolver = PBEA(problem, population_size=50, initial_population=ini_pop, total_function_evaluations=1000, indicator=preference_indicator, reference_point=ref_point)
    while evolver.continue_evolution():
        evolver.iterate()

    individuals2, objective_values2 = evolver.end()
    # should select small set of solutions to show to DM. For now we show all.
    true2 = plt.scatter(x=objective_values2[:,0], y=objective_values2[:,1], label="PBEA Front")
    plt.title(f"Fronts")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()


# TODO: 
# domination comparison for fitness/objective vectors
if __name__=="__main__":

   testZDTs()
   #testDTLZs()

   import cProfile
   #cProfile.run('testDTLZs()', "output.dat")

   import pstats
   from pstats import SortKey

   with open('output_time.txt', "w") as f:
       p = pstats.Stats('output.dat', stream=f)
       p.sort_stats('time').print_stats()

   with open('output_calls.txt', "w") as f:
       p = pstats.Stats('output.dat', stream=f)
       p.sort_stats('calls').print_stats()
