from typing import Dict, Type, Union, Tuple, Callable
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_tools.scalarization import SimpleASF
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_tools.utilities.quality_indicator import preference_indicator 
from desdeo_tools.utilities.distance_to_reference_point import distance_to_reference_point

# imports for testing TODO: remove
from desdeo_tools.utilities.quality_indicator import epsilon_indicator, epsilon_indicator_ndims 
from desdeo_emo.EAs.IBEA import IBEA
import matplotlib.pyplot as plt
from desdeo_problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

# this probably could be turned to a numpy broadcasting calculation
def preference_indicator2(reference_front: np.ndarray, front: np.ndarray, minasf, ref_point: np.ndarray, delta: float) -> float:
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
    # Rememeber np ones like ?
    #ref_front_asf = SimpleASF(np.ones_like(reference_front))
    front_asf = SimpleASF(np.ones_like(front))
    # this minasf just needs to be updated correctly .. 
    norm = front_asf(front, reference_point=ref_point) + delta - minasf
    # just to test this doesnt happen. Happens with dtlz2. though it probably shouldnt
    ##if norm < delta:
        #print(norm)
        #norm = delta # dunno if better this way
        #input()

    #print("minasdf", minasf)
    #print("norm", norm)
    return epsilon_indicator(reference_front, front)/norm



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
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        kappa: float = 0.5,
        indicator: Callable = preference_indicator,
        reference_point = None,
        delta: float = 0.1, 
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
        self.indicator = indicator
        self.reference_point = reference_point
        selection_operator = EnvironmentalSelection(self.population)
        self.selection_operator = selection_operator

def inter_dtlz():
    #problem_name = "DTLZ2" # seems work ok.
    problem_name = "DTLZ4" # this just starts going worse and worse 

    # test variables
    pop_s = 100
    iters = 3
    evals = 5000
    kappa = 0.05

    problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=pop_s, n_iterations=iters, n_gen_per_iter=100,total_function_evaluations=evals)
    while ib.continue_evolution():
        ib.iterate()
    # get the non dominated population
    individuals, objective_values = ib.end()
    print("ideal from IBEA approximation: ", ib.population.problem.ideal)

    # need to get the population
    ini_pop = ib.population
    # step 1: reference point. TODO: actually ask from DM
    delta = 0.1
    # step 2: local approximation
    evolver = PBEA(problem, interact=True, population_size=pop_s, initial_population=ini_pop, 
                   n_iterations=iters, n_gen_per_iter=100, total_function_evaluations=evals, 
                   indicator=preference_indicator2, kappa=kappa, delta=delta)
    
    print(evolver.delta)

    # hardcoded responses just to test
    dtlz4_response = np.asarray([[0.8, 0.6, 0.6],[0.35,0.7,0.55],[0.2,0.6,0.2]])
    # eli jos uusi refpoint huonompi kuin vanha, niin todnäk hajoaa..
    # koskee vain DTLZ, ei pysty toistamaan zdt

    # responses to use
    responses = dtlz4_response

    pref, plot = evolver.requests() # ask preference
    pref.response = pd.DataFrame([responses[0]], columns=pref.content['dimensions_data'].columns) # give preference
    pref, plot = evolver.iterate(pref) # iterate
    # achievement function
    # test like this
    #objectives = evolver.population.objectives
    t, objectives = evolver.end()
    d, ind  = distance_to_reference_point(objectives, responses[0]) # show best solution
    individuals1, objective_values1 = evolver.end()    
    #obj1 = evolver.population.objectives
    id1, obj1 = evolver.end() # turha, sama kuin ylempi..
    print("first iter",evolver._current_gen_count)

    evolver.delta = 0.01
    pref, plot = evolver.requests()
    pref.response = pd.DataFrame([responses[1]], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    # achievement function
    # test like this
    #objectives = evolver.population.objectives
    t, objectives = evolver.end()
    d2, ind2 = distance_to_reference_point(objectives,responses[1]) # show best solution
    individuals2, objective_values2 = evolver.end()
    #obj2 = evolver.population.objectives
    id2, obj2 = evolver.end()
    print("2nd run",evolver._current_gen_count)

    print(evolver.population.objectives)

    #TODO: possible solution
    # normalize ASF ? Make sure its correct
    # max_indicator !! ? if needed

    #evolver.delta = 0.02 # change delta
    #pref, plot = evolver.requests()
    #pref.response = pd.DataFrame([responses[2]], columns=pref.content['dimensions_data'].columns)
    #pref, plot = evolver.iterate(pref)
    ## achievement function
    #d3, ind3 = distance_to_reference_point(evolver.population.objectives, responses[2]) # show best solution
    #plot_obj3 = evolver.population.objectives
    #id3, obj3 = evolver.end()

    print(evolver.delta)
    print(evolver.n_iterations)
    print(evolver._current_gen_count)
    print(evolver._function_evaluation_count)

    #individuals3, obj_val = evolver.end()    
            
    # should select small set of solutions to show to DM. For now we show all.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(45,45)
    ax.scatter(objective_values[:,0],objective_values[:,1],objective_values[:,2], label="IBEA Front")
    ax.scatter(objective_values1[:,0], objective_values1[:,1], objective_values1[:,2], label="PBEA Front iter 1")
    ax.scatter(responses[0][0], responses[0][1], responses[0][2], label="Ref point 1")
    ax.scatter(obj1[ind][0], obj1[ind][1], obj1[ind][2], label="Best solution iteration 1")
    ax.scatter(objective_values2[:,0], objective_values2[:,1], objective_values2[:,2], label="PBEA Front iter 2")
    ax.scatter(responses[1][0], responses[1][1], responses[1][2], label="Ref point 2")
    ax.scatter(obj2[ind2][0], obj2[ind2][1], obj2[ind2][2], label="Best solution iteration 2")
    #ax.scatter(objective_values3[:,0], objective_values3[:,1], objective_values3[:,2], label="PBEA Front iter 1")
    #ax.scatter(responses[2][0], responses[2][1], responses[2][2], label="Ref point 3")
    #ax.scatter(obj3[ind3][0], obj3[ind3][1], obj3[ind3][2], label="Best solution iteration 3")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

def inter_zdt():
    #problem_name = "ZDT3" # seems work ok.
    problem_name = "ZDT1" # this just starts going worse and worse 

    # test variables
    pop_s = 32
    iters = 3
    evals = 5000
    kappa = 0.05


    problem = test_problem_builder(problem_name)
    # step 0. Let's start with rough approx
    ib = IBEA(problem, population_size=pop_s, n_iterations=iters, n_gen_per_iter=100,total_function_evaluations=2000, 
              indicator=epsilon_indicator)
    while ib.continue_evolution():
        ib.iterate()
    # get the non dominated population
    individuals, objective_values = ib.end()
    print("ideal from IBEA approximation: ", ib.population.problem.ideal)

    # need to get the population
    ini_pop = ib.population

    # step 1: reference point. TODO: actually ask from DM
    delta = 0.2

    # step 2: local approximation
    evolver = PBEA(problem, interact=True, population_size=pop_s, initial_population=ini_pop, 
                   n_iterations=iters, n_gen_per_iter=100, total_function_evaluations=evals, 
                   indicator=preference_indicator2, kappa=kappa, delta=delta)
    
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
    # achievement function. Dis loggiikka selvitä
    d, ind  = distance_to_reference_point(evolver.population.objectives, responses[0]) # show best solution
    individuals1, objective_values1 = evolver.end()    
    plot_obj1 = evolver.population.objectives
    print(evolver._current_gen_count)

    evolver.delta = 0.1 # change delta
    pref, plot = evolver.requests()
    pref.response = pd.DataFrame([responses[1]], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    # achievement function
    d2, ind2 = distance_to_reference_point(evolver.population.objectives,responses[1]) # show best solution
    individuals2, objective_values2 = evolver.end()
    plot_obj2 = evolver.population.objectives
    print(evolver._current_gen_count)

    evolver.delta = 0.01 # change delta
    pref, plot = evolver.requests()
    pref.response = pd.DataFrame([responses[2]], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    # achievement function
    d3, ind3 = distance_to_reference_point(evolver.population.objectives, responses[2]) # show best solution
    plot_obj3 = evolver.population.objectives

    print(evolver.delta)
    print(evolver.n_iterations)
    print(evolver._current_gen_count)
    print(evolver._function_evaluation_count)

    individuals3, objective_values3 = evolver.end()    
            
    print(evolver.population.fitness)
    
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
    #inter_dtlz()
        
    #import cProfile
    #cProfile.run('inter_zdt()', "output.dat")

    #import pstats
    #from pstats import SortKey

    #with open('output_time.txt', "w") as f:
    #    p = pstats.Stats('output.dat', stream=f)
    #    p.sort_stats('time').print_stats()

    #with open('output_calls.txt', "w") as f:
    #    p = pstats.Stats('output.dat', stream=f)
    #    p.sort_stats('calls').print_stats()
