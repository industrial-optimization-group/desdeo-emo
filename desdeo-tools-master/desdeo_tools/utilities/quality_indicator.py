
from numba import njit
import numba
import numpy as np
import hvwfg as hv

from desdeo_tools.scalarization import SimpleASF


@njit()
def epsilon_indicator(s1: np.ndarray, s2: np.ndarray) -> float:
    """ Computes the additive epsilon-indicator between two solutions.

    Args:
        s1 (np.ndarray): Solution 1. Should be an one-dimensional array.
        s2 (np.ndarray): Solution 2. Should be an one-dimensional array.

    Returns:
        float: The maximum distance between the values in s1 and s2.
    """
    eps = 0.0
    for i in range(s1.size):
        value = s2[i] - s1[i]
        if value > eps:
            eps = value
    return eps


@njit()
def epsilon_indicator_ndims(front: np.ndarray, reference_point: np.ndarray) -> list:
    """ Computes the additive epsilon-indicator between reference point and current one-dimensional vector of front.

    Args:
        front (np.ndarray): The front that the current reference point is being compared to.
            Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        reference_point (np.ndarray): The reference point that is compared. Should be one-dimensional array.

    Returns:
        list: The list of indicator values.
    """
    min_eps = 0.0
    eps_list = np.zeros((front.shape[0]), dtype=numba.float64)
    for i in np.arange(front.shape[0]):
        value = np.max(reference_point - front[i])
        if value > min_eps:
            eps_list[i] = value
    return eps_list



def preference_indicator(s1: np.ndarray, s2: np.ndarray, min_asf_value: float, ref_point: np.ndarray, delta: float) -> float:
    """ Computes the preference-based quality indicator.

    Args:
        s1 (np.ndarray): Solution 1. Should be an one-dimensional array.
        s2 (np.ndarray): Solution 2. Should be an one-dimensional array.
        ref_point (np.ndarray): The reference point should be same shape as front.
        min_asf_value (float): Minimum value of achievement scalarization of the reference_front. Used in normalization.
        delta (float): The spesifity delta allows to set the amplification of the indicator to be closer or farther 
            from the reference point. Smaller delta means that all solutions are in smaller range around the reference
            point.

    Returns:
        float: The maximum distance between the values in s1 and s2 taking into account 
            the reference point and spesifity.
    """
    s2_asf = SimpleASF(np.ones_like(s2))
    norm = s2_asf(s2, reference_point=ref_point) + delta - min_asf_value
    return epsilon_indicator(s1, s2) / norm


def hypervolume_indicator(front: np.ndarray, reference_point: np.ndarray) -> float:
    """ Computes the hypervolume-indicator between reference front and current approximating point.

    Args:
        front (np.ndarray): The front that is compared. Should be set of arrays, where the rows are the solutions and 
            the columns are the objective dimensions.
        reference_point (np.ndarray): The reference point that the current front is being compared to. Should be 1D array.

    Returns:
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    ref = np.asarray(reference_point, dtype='double') # hv.wfg needs datatype to be double
    fr = np.asarray(front, dtype='double')
    return hv.wfg(fr, ref)


if __name__=="__main__":

    po_front = np.asarray([[1.0,0],[0.5,0.5], [0,1.0], [2, -1], [0,0]])
    sol1 = [4, 4] # cant be better than po front, min is zero
    sol = np.asarray(sol1)
    ref = np.asarray([0.7, 0.3])

    print("eps indi value")
    print(epsilon_indicator(po_front[0], sol))
    print(epsilon_indicator(po_front[1], sol))
    print(epsilon_indicator(po_front[2], sol))
    print(epsilon_indicator(po_front[3], sol))
    print(epsilon_indicator(po_front[4], sol))
    print("ndims")
    print(epsilon_indicator_ndims(po_front, sol))
    print(hypervolume_indicator(po_front, sol))
    print("pref")
    print(preference_indicator(po_front[1], sol, 0.1, ref, 0.1))
