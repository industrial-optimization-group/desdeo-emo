import numpy as np
from typing import ClassVar, List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population

# TODO: fix the init to make sense

class EnvironmentalSelection(SelectionBase):
    def __init__(self, ea):
        self.algo = ea
        self.population = ea.population
        self.worst_fit = None
               

    def do(self) -> int:
        
        while (self.population.pop_size < self.population.individuals.shape[0]):
       # choose individual with smallest fitness value with environmentalSelection

            worst_index = self.enviSel(self.population)
            poplen = self.population.individuals.shape[0]
            for i in range(poplen):
                if worst_index != i:
                    if self.algo.reference_point is not None: 
                        self.population.fitness[i] -= -np.exp(-self.algo.indicator(self.population.objectives[i], self.population.objectives[worst_index], self.algo.min_asf_value, self.algo.reference_point, self.algo.delta) / self.algo.kappa)
                    else:
                        self.population.fitness[i] += np.exp(-self.algo.indicator(self.population.objectives[i], self.population.objectives[worst_index]) / self.algo.kappa)
       
                # remove the worst individual 
            self.population.delete(worst_index)

        return 0


    def enviSel(self, pop: Population):
        fit_min = np.Inf 
        pop_size = pop.individuals.shape[0]
        
        # 1. choose a point x* in P with the smallest fitness value
        for i in range(pop_size):
            if pop.fitness[i][0] < fit_min: 
                fit_min = pop.fitness[i][0]
                self.worst_fit = i 

        return self.worst_fit 

