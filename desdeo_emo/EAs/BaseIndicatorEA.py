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






# tämä uudempi versio
# 13627515   10.910    0.000   10.910    0.000 /home/jp/devaus/tyot/DESDEO/desdeo-emo/desdeo_emo/EAs/BaseIndicatorEA.py:28(epsilon_indicator)
#@njit() 
def epsilon_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """
    This now assumes reference_front and front are same dimensions
    """
    eps = 0.0
    for i in range(reference_front.size):
        value = front[i] - reference_front[i]
        if value > eps:
            eps = value
    return eps

# TODO: more testing required
# TODO: testaa onkop jotain rikki, kun tuntuu että jää pari huonoa ratkaisua kellumaan.. Ehkä vika uudessa envi sel?
# lyhyt on nopeampi, mutta tekeekö ihan oikein asiat?
# dtlz1 näytti lyhyen ratkaisu myös paremmaltakin ? :o

# fastest without using numba..
#  calls     tot time           cum time
# 13636377   17.907    0.000   19.280    0.000 /home/jp/devaus/tyot/DESDEO/desdeo-emo/desdeo_emo/EAs/BaseIndicatorEA.py:26(epsilon_indicator)
def epsilon_indicator_long(reference_front: np.ndarray, front: np.ndarray) -> float:
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
                    tour_select(population.fitness[:, 0], 2), # don't know if this fixes the issue
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
                self.population.fitness[i] += np.exp(-epsilon_indicator(self.population.objectives[i], self.population.objectives[worst_index]) / 0.05)
                #self.population.fitness[i] += np.exp(-epsilon_indicator_long([self.population.objectives[i]],[self.population.objectives[worst_index]]) / 0.05)

            # remove the worst individula 
            self.population.delete(selected)

            # update the fitness values of remaining individuals. Here we should use worst index.. also would probably make it faster.
           # self._fitness_assignment()
        # check termination
        if (self._function_evaluation_count >= self.total_function_evaluations):
            # just to stop the iteration. TODO: do it better
            self.total_function_evaluations = 1
            return

        # perform binary tournament selection. in these steps 5 and 6 we give offspring to the population and make it bigger. kovakoodataan tämä nytten, mietitään myöhemmin sitten muuten.
        chosen = binary_tournament_select(self.population)
        #print(f" chosen pop ka {np.mean(chosen)} chosen pop kh {np.std(chosen)}")

        # variation, call the recombination operators
        offspring = self.population.mate(mating_individuals=chosen)
        self.population.add(offspring)

        self._current_gen_count += 1
        #print(self._current_gen_count)
        self._gen_count_in_curr_iteration += 1
        #print(f"gen count curr ite {self._gen_count_in_curr_iteration}")
        self._function_evaluation_count += offspring.shape[0]
        #print(self._function_evaluation_count)


    # need to implement the enviromental selection. Only calls it from selection module
    def _select(self) -> list:
        return self.selection_operator.do(self.population)

    #implements fitness computing. 
    # TODO: trouble of calling different indicators with the design since indicators are just functions. Let's cross that bridge when we come to it.
    def _fitness_assignment(self):
        population = self.population
        pop_size = population.individuals.shape[0]
        pop_width = population.fitness.shape[1]

        #print(f"pop fitness ka: {np.mean(population.fitness)}, pop fitness kh: {np.std(population.fitness)}")

        fit_comp = np.zeros([pop_size, pop_size])
        # TODO: population.fitness[i] could be the issue, since it has objective number of values but we put the indicator value to all of them.
        for i in range(pop_size):
            #population.fitness[i] = [0]*pop_width # 0 all the fitness values. 
            population.fitness[i] = 0 # zero only first index, we are only using that in the calculations 
            for j in range(pop_size):
                if j != i:
                    population.fitness[i] += -np.exp(-epsilon_indicator(population.objectives[i], population.objectives[j]) / 0.05)
                    #population.fitness[i] += -np.exp(-epsilon_indicator_long([population.objectives[i]],[population.objectives[j]]) / 0.05)

