from typing import TYPE_CHECKING

from pyRVEA.Selection.NSGAIII_select import NSGAIII_select
from pyRVEA.EAs.baseEA import BaseDecompositionEA
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors

import numpy as np

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population


class NSGAIII(BaseDecompositionEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    [description]
    """
    def set_params():
        pass
    
    def select():
        pass
    
    def _run_interruption():
        return