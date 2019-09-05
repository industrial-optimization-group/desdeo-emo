from abc import ABC, abstractmethod
import numpy as np
from typing import List


class SelectionBase(ABC):
    """The base class for the selection operator.
    """
    @abstractmethod
    def do(self, fitness: np.ndarray, *args) -> List[int]:
        """Use the selection operator over the given fitness values. Return the indices
            individuals with the best fitness values according to the operator.

        Parameters
        ----------
        fitness : np.ndarray
            Fitness of the individuals from which the next generation is to be selected.

        Returns
        -------
        List[int]
            The list of selected individuals
        """
        pass
