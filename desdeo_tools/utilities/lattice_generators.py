"""A file to contain different kinds of lattice generation algorithms.
"""

import numba
import numpy as np


@numba.njit()
def fibonacci_sphere(samples: int = 1000) -> np.ndarray:
    """Generate a very even lattice of points on a 3d sphere using the fibonacci sphere
    or fibonacci spiral algorithm.

    Args:
        samples (int, optional): Number of points to be generated. Defaults to 1000.

    Returns:
        np.ndarray: The lattice of points as a 2-D (samples, 3) numpy array.
    """
    points = np.zeros((samples, 3), dtype=np.float_)
    phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians

    for i in range(samples):
        points[i, 1] = 1 - (i / (samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - points[i, 1] ** 2)  # radius at y

        theta = phi * i  # golden angle increment

        points[i, 0] = np.cos(theta) * radius
        points[i, 2] = np.sin(theta) * radius
    return points
