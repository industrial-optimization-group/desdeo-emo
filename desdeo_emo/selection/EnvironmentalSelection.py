import numpy as np
from typing import ClassVar, List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population

# TODO: fix the init to make sense

class EnvironmentalSelection(SelectionBase):
    def __init__(self):
        pass 

    def do(self, pop) -> int:
        return np.argmin(pop.fitness, axis=0)[0]

