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
        
        # this seems really bit faster but I am not completely sure its better/works correctly.
        # oikea huoli siitä ettei toimi oikein, jää tällä huoonoja ratkaisuja roikkumaan
        # TOOD: make sure it works in all cases
   #  calls   tot time           cum time
#    24667    0.059    0.000    0.344    0.000 /home/jp/devaus/tyot/DESDEO/desdeo-emo/desdeo_emo/selection/EnvironmentalSelection.py:14(do)
#        fit_min = np.min(pop.fitness)
        #print("fit_min", fit_min)
#        worst_fit = np.where(np.any(fit_min == pop.fitness, axis = 1))
        #print("worst fit", worst_fit[0])
#        selection = worst_fit[0]
        
#       # vanha
        for i in range(pop_size):
            #print("f vektor",pop.fitness[i][0])
            temp = pop.fitness[i][0] # sum might not be a good idea either, maybe min is better idea
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
