import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class EnvironmentalSelection(SelectionBase):
    def __init__(self, pop: Population):
        print("enviselect init")
                

    # dunno if my OOP makes sense again
    # TODO: see if fitness vector makes this a problem..
    def do(self, pop: Population) -> List[int]:
        # 1. choose a point x* in P with the smallest fitness value
        fit_min = np.Inf 
        worst_fit = None
       
        pop_size = pop.individuals.shape[0]
        #print("envsel pop size", pop_size)
        #print(pop.fitness[0])
        ##print(sum(pop.fitness[0]))
        # TODO: step 3.1 is not there. We need to find dominated x*
        for i in range(pop_size):
            temp = np.min(pop.fitness[i]) # sum might not be a good idea either, maybe min is better idea
            if temp <= fit_min: # no mitä nyt tehdään kun on fitness taulukko tosiaan (100,2) eli tässä kaksi arvoa. Any might not do the right thing
                fit_min = temp
                worst_fit = i 
        # 2. remove x* from the population
        #print("worst fit", worst_fit)
        # for returning data in correct form. Clumsy way, fix it
        selection = np.asarray([], dtype=int)
        for i in range(pop_size):
            if (i == worst_fit):
                selection = np.append(selection, i)

        return selection        
