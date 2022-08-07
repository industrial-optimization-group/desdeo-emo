import numpy as np

from typing import Tuple, Type
from desdeo_tools.scalarization import SimpleASF


def distance_to_reference_point(obj: np.ndarray, reference_point: np.ndarray) -> Tuple:
    """Computes the closest solution to a reference point using achievement scalarizing function.

    Args:
        obj (np.ndarray): Array of the solutions. Should be 2d-array.
        reference_point (np.ndarray): The reference point array. Should be one dimensional array.

    Returns:
        Tuple: Returns a tuple containing the closest solution to a reference point and the index of it in obj.
    """
    asf = SimpleASF(np.ones_like(obj))
    res = asf(obj, reference_point=reference_point)
    return np.min(res), np.argmin(res)
