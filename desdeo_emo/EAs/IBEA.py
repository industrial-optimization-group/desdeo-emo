from typing import Dict, Type, Union, Tuple, Callable
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA 
from desdeo_tools.utilities.quality_indicator import epsilon_indicator

# TODO: remember to sort import

# imports for testing. TODO: remove later
import matplotlib.pyplot as plt
from desdeo_problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder







class IBEA(BaseIndicatorEA):
    """Python Implementation of IBEA. 

    Most of the relevant code is contained in the super class. This class just assigns
    the EnviromentalSelection operator to BaseIndicatorEA.

    Parameters
    ----------
    problem: MOProblem
        The problem class object specifying the details of the problem.
    population_size : int, optional
        The desired population size, by default None, which sets up a default value
        of population size depending upon the dimensionaly of the problem.
    population_params : Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population : Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    a_priori : bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact : bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    n_iterations : int, optional
        The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter : int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations : int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    kappa : float, optional
        Fitness scaling value for indicators. By default 0.05.
    indicator : Callable, optional
        Quality indicator to use in indicatorEAs. By default in IBEA this is additive epsilon indicator.

    """
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
        kappa: float = 0.05, # fitness scaling ratio
        indicator: Callable = epsilon_indicator, # default indicator is epsilon_indicator
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
            kappa=kappa,
        )
        self.indicator = indicator
        self.kappa = kappa
        selection_operator = EnvironmentalSelection()
        self.selection_operator = selection_operator

    
    
    def _fitness_assignment(self):
        for i in range(self.population.individuals.shape[0]):
            self.population.fitness[i] = 0 # 0 all the fitness values. 
            for j in range(self.population.individuals.shape[0]):
                if j != i:
                   self.population.fitness[i] += -np.exp(-self.indicator(self.population.objectives[i], self.population.objectives[j]) / self.kappa)


    def _update_fitness(self, worst_index):
        for i in range(self.population.individuals.shape[0]):
            if worst_index != i:
                self.population.fitness[i] += np.exp(-self.indicator(self.population.objectives[i], 
                    self.population.objectives[worst_index]) / self.kappa)



"""
25 000 evals
ZDT1 works
ZDT2 works 
ZDT3 works
ZDT4 works
ZDT6 works
"""
def testZDTs():
    problem_name = "ZDT1" # needs 30,100. ZDT1 seems to converge even with about 2000 total_function_evaluations
    #problem_name = "ZDT2" # seems work ok.
    #problem_name = "ZDT6" # this just starts going worse and worse 
    # doesn't work properly with ZDT4... atleast saves too many bad solutions..

    problem = test_problem_builder(problem_name)
    evolver = IBEA(problem,indicator=epsilon_indicator, population_size=50, n_iterations=10,n_gen_per_iter=100, 
                   total_function_evaluations=7000, kappa=0.05)
    
    #print("starting front", evolver.population.objectives[0::10])
    while evolver.continue_evolution():
        evolver.iterate()

    # evolver.iterate() # for some reason this stops at 10100 iters and won't listen our termination, hence bad results
    #front_true = evolver.population.objectives
    individuals, front_true = evolver.end()
    #print(front_true[0::10])

    true = plt.scatter(x=front_true[:,0], y=front_true[:,1], label="True Front")
    plt.title(f"Fronts obtained with various algorithms on the problem")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()


"""
25 000 evals
DTLZ1 works with my eps_indi
DTLZ2 works, i guess could have more points in the middle and less on the sides
DTLZ3 doesnt really work. Can IBEA even solve this problem? 
DTLZ4 works, i guess could have more points in the middle and less on the sides
DTLZ5 works
DTLZ6 works
DTLZ7 works, could have more points in the middle, maybe with more population members?
"""
def testDTLZs():
    #problem_name = "DTLZ1" 
    #problem = test_problem_builder(problem_name, n_of_variables=7, n_of_objectives=3)
        
    problem_name = "DTLZ2" # seems to work okay?, even with low total_function_evaluations. po sols are not that even in places tho.
    problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    #problem_name = "DTLZ4" # seems to work okay?, even with low total_function_evaluations. po sols are not that even in places tho.
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
    
    #problem_name = "DTLZ6" # does not do that good.. mean and std get low but then they start oscillating
    #problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

    #problem_name = "DTLZ7" # this looks pretty good, same as dtlz6 for the mean and std 
    #problem = test_problem_builder(problem_name, n_of_variables=22, n_of_objectives=3)

    evolver = IBEA(problem, population_size=50, n_iterations=10, n_gen_per_iter=100, total_function_evaluations=7500)
    
    #print("starting front", evolver.population.objectives[0::10])
    while evolver.continue_evolution():
        evolver.iterate()
        
    individuals, front_true = evolver.end()
    #print(result)
    #front_true = result[1]
    #print(front_true[0::10])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30,45)
    ax.scatter(front_true[:,0],front_true[:,1],front_true[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



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
