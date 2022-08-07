"""Module which handles polytopes."""

import itertools
import numpy as np

from typing import Optional
from scipy.optimize import linprog


def inherently_nondominated(A: np.ndarray, epsilon: Optional[float] = 1e-06, method: Optional[str] = "highs") -> bool:
    """Check if a polytope is inherently nondominated:
    A polytope is inherently nondominated iff the polytope does not dominate itself.

    Args:
        A (np.ndarray): A polytope to be checked.
        epsilon (Optional[float], optional): precision parameter, see polytope_dominates for further details.
            Defaults to 1e-6.
        method (Optional[str], optional): Algorithm used to solve the optimization problems. Defaults to 'highs'.

    Returns:
        bool: is the given set inherently nondominated.
    """
    return not polytope_dominates(A, A, epsilon, method)


def polytope_dominates(
    k1: np.ndarray,
    k2: np.ndarray,
    epsilon: Optional[float] = 1e-6,
    method: Optional[str] = "highs"
) -> bool:
    """
    Check if polytope p(k1) dominates polytope p(k2) with epsilon certainty
    by solving linear optimization problems [min_x c^T*x] using linprog from scipy.optimize.

    Args:
        k1 (np.ndarray): Corners of first polytope
        k2 (np.ndarray): Corners of seconds polytope
        epsilon (Optional[float], optional): precision parameter. Defaults to 1e-6
        method (Optional[str], optional): Algorithm used to solve the optimization problems.
            Defaults to 'highs'. See scipy.optimize.linprog for further details.

    Returns:
        bool: Does polytope p(k1) dominate polytope p(k2).
    """
    k1 = np.atleast_2d(k1)
    k2 = np.atleast_2d(k2)
    a, k = k1.shape
    b = k2.shape[0]

    # First optimization problem
    coef = np.hstack((np.ones(1), np.zeros(a+b)))
    lower_bounds = np.hstack((np.array([None]), np.zeros(a+b)))
    upper_bounds = np.hstack((np.array([None]), np.ones(a+b)))
    bounds = np.vstack((lower_bounds, upper_bounds)).T

    # Constructing the matrix A_ub in the constraint A_ub x <= b_ub
    A1 = np.hstack((-np.ones((k,1)), k1.T, -k2.T))
    A2 = np.hstack((np.zeros((1, 1)), np.ones((1,a)), np.zeros((1,b))))
    A3 = np.hstack((np.zeros((1, 1)), np.zeros((1,a)), np.ones((1,b))))
    A_ub = np.vstack((A1, np.zeros((2,a+b+1))))
    b_ub = np.zeros(k+2)

    A_eq = np.vstack((np.zeros((k,a+b+1)), A2, A3)) # Add k rows for correct size, will be ignored
    b_eq = np.hstack((np.zeros(k), np.ones(2))) 

    res = linprog(coef, A_ub, b_ub, A_eq, b_eq, bounds, method = method)

    if not res['success']: 
        print("unsuccessful optimization in first problem.")
    if res['fun'] < -epsilon: 
        return True

    if (abs(res['fun']) <= epsilon):
        lower_bounds = np.zeros(a+b)
        upper_bounds = np.ones(a+b)
        bounds = np.vstack((lower_bounds, upper_bounds)).T
        
        coef = np.hstack(([np.sum(k1, axis = 1), -np.sum(k2, axis = 1)]))
        A1 = np.hstack((np.ones((1,a)), np.zeros((1,b))))
        A2 = np.hstack((np.zeros((1,a)), np.ones((1,b))))
        A3 = np.hstack((k1.T, -k2.T))

        A_eq = np.vstack((A1, A2, np.zeros((k, a+b))))
        b_eq = np.hstack((np.ones(2), np.zeros(k)))

        A_ub = np.vstack((np.zeros((2, a+b)), A3))
        b_ub = np.zeros((k+2, 1))

        res = linprog(coef, A_ub, b_ub, A_eq, b_eq, bounds, method = method)
 
        if not res['success']: 
            print("unsuccessful optimization in second problem.")
        if res['fun'] < -epsilon: 
            return True
    return False


def generate_polytopes(simplices: np.ndarray) -> np.ndarray:
    """Generate polytopes from an array of indices which form simplices

    Args:
        arr (np.ndarray): An array of indices which form simplices.
            In PAINT this is the array of simplices which form the Delaunay triangulation.

    Returns
        np.ndarray: An array of indices which form the polytopes
            that are generated from the given array. If a polytope has fewer
            outcomes than there are columns in the given array the first value of
            the row representing the polytope is repeated until the lengths match.
    """
    simplices = np.sort(simplices)
    a,b = simplices.shape
    k = np.max(simplices) # point count
    F = (np.ones((b,k+1)) *np.arange(k+1)).T
    for i in range(a):
        for j in range(2,b+1):
            # All combinations of size j from row i
            addition = np.array(list(itertools.combinations(simplices[i], j)))
            # Add all combinations to F, so that we repeat the value at index 0 until enough values
            chunks = addition[:,0].shape[0] # How many chunks to split into
            repeated = np.split(np.repeat(addition[:,0], b-j), chunks) # Repeat the values at index 0
            addition = np.hstack((addition, repeated)) # Add the repeated values
            F = np.vstack((F, addition)) # Add new rows to F
    # F(F(:,1)==0,:) = []; ?
    return np.unique(F, axis = 0).astype(int)
