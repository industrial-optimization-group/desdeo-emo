import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class EnvironmentalSelection(SelectionBase):
    def __init__(self, pop: Population):
        #self.pop_size = pop.individuals.shape[0]
        print("enviselect init")
                

    # TODO: do better
    # finds the index of an individual with the smallest fitness value
    def do(self, pop: Population) -> int:
        fit_min = np.Inf 
        worst_fit = None
        pop_size = pop.individuals.shape[0]
        
        # 1. choose a point x* in P with the smallest fitness value
        for i in range(pop_size):
            temp = pop.fitness[i][0] 
            if temp < fit_min: 
                fit_min = temp
                worst_fit = i 

        return worst_fit 
