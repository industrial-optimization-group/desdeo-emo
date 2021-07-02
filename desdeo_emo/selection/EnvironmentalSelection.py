import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


# only temporarily here until desdeo_tools get fixed 
#@njit 
def epsilon_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
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




class EnvironmentalSelection(SelectionBase):
    def __init__(self, pop: Population):
        print("enviselect init")
                

    # dunno if my OOP makes sense again
    def do(self, pop: Population) -> List[int]:
        # 1. choose a point x* in P with the smallest fitness value
        fit_min = np.Inf 
        worst_fit = None
       
        pop_size = pop.individuals.shape[0]
        #print("envsel pop size", pop_size)
        #print(pop.fitness[0])
        ##print(sum(pop.fitness[0]))
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
