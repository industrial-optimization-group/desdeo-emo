import numpy as np


def get_2D_version(x, pi1, pi2):
    """
    Project n > 2 dimensional vector to 2-dimensional space

    Args:
        x (np.ndarray): A given vector to project to 2-dimensional space

    Returns:
        np.ndarray: A 2-dimensional vector
    """
    if x.shape[1] <= 2:
        return x
    left = np.divide(np.dot(x, pi1), np.sum(pi1))  # Left side of vector
    right = np.divide(np.dot(x, pi2), np.sum(pi2))  # Right side of vector
    return np.hstack((left, right))


# X1 is a matrix or array, size n x number of design variables
# x2 is a array, size 1 x number of design variables
def euclidean_distance(x1, x2):
    """
    Returns the euclidean distance between x1 and x2.
    """
    if x1 is None or x2 is None:
        print("euclidean distance supplied with nonetype")
        return None
    return np.linalg.norm(x1 - x2, axis=-1)


def get_random_angles(n):
    return np.random.rand(n, 1) * 2 * np.pi


def between_lines_rooted_at_pivot(x, pivot_loc, loc1, loc2) -> bool:
    """

    Args:
        x (np.ndarray): 2D point to check
        pivot_loc: attractor on boundary of circle
        loc1: another point on boundary of circle
        loc2: another point on boundary of circle

    Returns:
        bool: true if x on different side of line defined by pivot_loc and loc1, \
        compared to the side of the line defined by pivot_loc and loc2.
        If x is also in the circle, then x is betweeen the two lines if return is true.
    """
    t = False
    d1 = (x[0] - pivot_loc[0]) * (loc1[1] - pivot_loc[1]) - (x[1] - pivot_loc[1]) * (loc1[0] - pivot_loc[0])
    d2 = (x[0] - pivot_loc[0]) * (loc2[1] - pivot_loc[1]) - (x[1] - pivot_loc[1]) * (loc2[0] - pivot_loc[0])

    if d1 == 0:
        t = True
    elif d2 == 0:
        t = True
    elif np.sign(d1) != np.sign(d2):
        t = True

    return t


def assign_design_dimension_projection(n_variables, vary_sol_density):
    """
    if more than two design dimensions in problem, need to assign
    the mapping down from this higher space to the 2D version
    which will be subsequantly evaluated
    """
    if n_variables <= 2:
        return None, None
    mask = np.random.permutation(n_variables)
    if vary_sol_density:
        diff = np.random.randint(n_variables - 1)
        mask = mask[:diff]  # Take the diff first elements
    else:
        half = int(np.ceil(n_variables / 2))
        mask = mask[:half]  # Take half first elements
    pi1 = np.zeros(n_variables, dtype=bool)
    pi1[mask] = True
    pi2 = pi1
    pi2 = np.ones(n_variables, dtype=bool)
    pi2[mask] = False

    return pi1, pi2
