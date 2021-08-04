from typing import Dict, Type, Union, Tuple, Callable
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_emo.selection.tournament_select import tour_select
from desdeo_tools.scalarization import SimpleASF
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA

# need to add preference_indicator
#from desdeo_tools.utilities.quality_indicator import preference_indicator 


# imports for testing TODO: remove
from desdeo_tools.utilities.quality_indicator import epsilon_indicator 
from desdeo_emo.EAs.IBEA import IBEA
import matplotlib.pyplot as plt
from desdeo_problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

# TODO: test, make better
# find proper place, would be neat if it could easily be put in place of eps indi.. to desdeo_tools again we go i think
# Ip(y, x) =) Ie(y,x) / s(g, f(x), delta)


def preference_indicator(reference_front: np.ndarray, front: np.ndarray, ref_point: np.ndarray, delta: float) -> float:
    """ Computes the preference-based quality indicator.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        front (np.ndarray): The front that is compared. Should be one-dimensional array with the same shape as
        reference_front.
        ref_point (np.ndarray): The reference point should be same shape as front.
        delta (float): The spesifity delta allows to set the amplification of the indicator to be closer or farther 
        from the reference point. Smaller delta means that all solutions are in smaller range around the reference
        point.

    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives taking into account the reference point given and spesifity.
    """
    ref_front_asf = SimpleASF(reference_front)
    front_asf = SimpleASF(front)
    norm = front_asf(front, reference_point=ref_point) + delta - np.min(ref_front_asf(reference_front, reference_point=ref_point))
    return epsilon_indicator(reference_front, front)/norm


# temp will be shown to the DM, hence needed !
# index might be useful depending on what I do..
# the actual objective vector might be useful with showing the dm..

# index could be used instead of latter two

# TODO: make better. Bit better..
def distance_to_reference_point(obj, asf, reference_point):
    d = (np.Inf, 1)
    for i, k in enumerate(obj):
        i_d = asf(k, reference_point=reference_point)
        if i_d < d[0]:
            d = i_d, i

    return d

#  need to figure out how to return part of the pop / the best solution by the asf for the DM to give new preference. 
# As now, the evolver.end() returns the non dominated which is good for last iteration but it ends the iteration also..
# one solution is start another PBEA with this new nondom population but that doesnt sound that good of a solution.
#
#

# for start we can implement stuff here too
class PBEA(BaseIndicatorEA):
    """Python Implementation of PBEA. 

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
        Quality indicator to use in indicatorEAs. For PBEA this is preference based quality indicator.
    reference_point : np.ndarray
        The reference point that guides the PBEAs search.
    delta : float, optional
        Spesifity for the preference based quality indicator. 

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
        #print(self.reference_point)
        selection_operator = EnvironmentalSelection(self.population)
        self.selection_operator = selection_operator

        print("using PBEA")


        # PBEA needs to do the step 3. Aka show some of the solutions


def inter_zdt():
    #problem_name = "ZDT3" # seems work ok.
    problem_name = "ZDT1" # this just starts going worse and worse 

    # test variables
    pop_s = 32
    iters = 3
    evals = 2000
    kappa = 0.05


    problem = test_problem_builder(problem_name)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=pop_s, n_iterations=iters, n_gen_per_iter=100,total_function_evaluations=evals)
    while ib.continue_evolution():
        ib.iterate()
    # get the non dominated population
    individuals, objective_values = ib.end()
    print("ideal from IBEA approximation: ", ib.population.problem.ideal)

    # need to get the population
    print(ib.return_pop())
    ini_pop = ib.return_pop()

    # step 1: reference point. TODO: actually ask from DM
    delta = 0.1

    # step 2: local approximation
    evolver = PBEA(problem, interact=True, population_size=pop_s, initial_population=ini_pop, 
                   n_iterations=iters, n_gen_per_iter=100, total_function_evaluations=evals, 
                   indicator=preference_indicator, kappa=kappa, delta=delta)
    
    print(evolver.delta)

    # hardcoded responses just to test
    # desdeo's logic doesnt make yet sense so this won't work
    zdt1_response = np.asarray([[0.55,0.5], [0.40,0.45], [0.30, 0.40]]) 
    #zdt1_test = np.asarray([[0.6,0.8], [0.55,0.6], [0.50, 0.50]]) 
    #zdt6_response = np.asarray([[0.65,1.0], [0.5,0.9], [0.3, 0.85]]) 

    # responses to use
    #responses = zdt6_response 
    responses = zdt1_response 
    #responses = zdt1_test

    pref, plot = evolver.requests() # ask preference
    pref.response = pd.DataFrame([responses[0]], columns=pref.content['dimensions_data'].columns) # give preference
    pref, plot = evolver.iterate(pref) # iterate
    # achievement function
    asf = SimpleASF(evolver.population.objectives)
    d, ind  = distance_to_reference_point(evolver.population.objectives, asf, responses[0]) # show best solution
    individuals1, objective_values1 = evolver.end()    
    plot_obj1 = evolver.return_pop().objectives
    print(evolver._current_gen_count)

    pref, plot = evolver.requests()
    pref.response = pd.DataFrame([responses[1]], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    # achievement function
    asf = SimpleASF(evolver.population.objectives)
    d2, ind2 = distance_to_reference_point(evolver.population.objectives, asf, responses[1]) # show best solution
    individuals2, objective_values2 = evolver.end()    
    plot_obj2 = evolver.return_pop().objectives
    print(evolver._current_gen_count)

    evolver.delta = 0.03 # change delta
    pref, plot = evolver.requests()
    pref.response = pd.DataFrame([responses[2]], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    # achievement function
    asf = SimpleASF(evolver.population.objectives)
    d3, ind3 = distance_to_reference_point(evolver.population.objectives, asf, responses[2]) # show best solution
    plot_obj3 = evolver.return_pop().objectives

    print(evolver.delta)
    print(evolver.n_iterations)
    print(evolver._current_gen_count)
    print(evolver._function_evaluation_count)

    individuals3, objective_values3 = evolver.end()    
            
    # should select small set of solutions to show to DM. For now we show all.
    plt.scatter(x=objective_values[:,0], y=objective_values[:,1], label="IBEA Front")
    plt.scatter(x=objective_values1[:,0], y=objective_values1[:,1], label="PBEA Front iteration 1")
    plt.scatter(x=objective_values2[:,0], y=objective_values2[:,1], label="PBEA Front iteration 2")
    plt.scatter(x=objective_values3[:,0], y=objective_values3[:,1], label="PBEA Front at last iteration")
    plt.scatter(x=responses[0][0], y=responses[0][1],  label="Ref point 1")
    plt.scatter(x=responses[1][0], y=responses[1][1], label="Ref point 2")
    plt.scatter(x=responses[2][0], y=responses[2][1], label="Ref point 3")
    plt.scatter(x=plot_obj1[ind][0], y=plot_obj1[ind][1], label="Best solution iteration 1")
    plt.scatter(x=plot_obj2[ind2][0], y=plot_obj2[ind2][1], label="Best solution iteration 2")
    plt.scatter(x=plot_obj3[ind3][0], y=plot_obj3[ind3][1], label="Best solution iteration 3")
    plt.title(f"Fronts")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()


if __name__=="__main__":

    inter_zdt()

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
