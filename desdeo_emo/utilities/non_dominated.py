from typing import Tuple

import numpy as np
from desdeo_tools.utilities import dominates
from numba import njit


@njit()
def check_domination(
    new: np.ndarray, archive: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Checks if newly evaluated and archived solutions are non_dominated.

    Args:
        new (np.ndarray): Newly evaluated solution's fitness values
             (should be a 2D array).
        archive (np.ndarray): Archive of non-dominated solutions
             (also a 2D array, must be mutually non-dominated).

    Raises:
        ValueError: If the new array is not a 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of indices that are
             non-dominated in the new and archive arrays respectively.
    """
    if new.ndim != 2:
        raise ValueError("New array should have 2 dimensions.")
    non_dominated_in_new = np.ones(len(new), dtype=np.bool_)
    non_dominated_in_archive = np.ones(len(archive), dtype=np.bool_)
    for i in range(len(new)):
        for j in range(len(archive)):
            if not non_dominated_in_archive[j]:
                continue
            if dominates(archive[j], new[i]):
                non_dominated_in_new[i] = False
                break  # No need to check with the rest of the archive
            if dominates(new[i], archive[j]):
                non_dominated_in_archive[j] = False
    return (non_dominated_in_new, non_dominated_in_archive)
