import numpy as np
from pygmo import fast_non_dominated_sorting as nds
from warnings import warn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyRVEA.allclasses import ReferenceVectors


def NSGAIII_select(
    fitness: list,
    vectors: "ReferenceVectors",
    ideal_point: list = None,
    worst_point: list = None,
    extreme_points: list = None,
    n_survive: int = None,
):
    # Calculating fronts and ranks
    fronts, dl, dc, rank = nds(fitness)
    non_dominated = fronts[0], fronts
    # Calculating worst points
    worst_of_population = np.amax(fitness, axis=0)
    worst_of_front = np.max(F[non_dominated, :], axis=0)
    extreme_points = get_extreme_points_c(
        F[non_dominated, :], ideal_point, extreme_points=extreme_points
    )
    nadir_point = get_nadir_point(
        extreme_points, ideal_point, worst_point, worst_of_population, worst_of_front
    )
    # Finding individuals in first 'n' fronts
    selection = np.asarray([], dtype=int)
    for front_id in len(fronts):
        if len(np.concatenate(fronts[:front_id+1])) < n_survive:
            continue
        else:
            selection = np.concatenate(fronts[:front_id+1])
            break
    last_front_id = front_id



def get_extreme_points_c(F, ideal_point, extreme_points=None):
    """Taken from pymoo"""
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]
    return extreme_points


def get_nadir_point(
    extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population
):
    LinAlgError = np.linalg.LinAlgError
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        if (
            not np.allclose(np.dot(M, plane), b)
            or np.any(intercepts <= 1e-6)
            or np.any(nadir_point > worst_point)
        ):
            raise LinAlgError()

    except LinAlgError:
        nadir_point = worst_of_front

    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]
    return nadir_point
