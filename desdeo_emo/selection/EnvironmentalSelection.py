import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population

# TODO: fix the init to make sense

class EnvironmentalSelection(SelectionBase):
    def __init__(self, pop: Population):
        self.worst_fit = None
                

    # TODO: do better
    # finds the index of an individual with the smallest fitness value
    def do(self, pop: Population) -> int:
        fit_min = np.Inf 
        pop_size = pop.individuals.shape[0]
        
        # 1. choose a point x* in P with the smallest fitness value
        for i in range(pop_size):
            if pop.fitness[i][0] < fit_min: 
                fit_min = pop.fitness[i][0]
                self.worst_fit = i 

        return self.worst_fit 
