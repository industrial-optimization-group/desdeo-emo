import numpy as np
from pygmo import fast_non_dominated_sorting as nds
from warnings import warn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrvea.allclasses import ReferenceVectors


def NSGAIII_select(
    fitness: list,
    ref_dirs: "ReferenceVectors",
    ideal_point: list = None,
    worst_point: list = None,
    extreme_points: list = None,
    n_survive: int = None,
):
    # Calculating fronts and ranks
    fronts, dl, dc, rank = nds(fitness)
    non_dominated = fronts[0]

    # Calculating worst points
    worst_of_population = np.amax(fitness, axis=0)
    worst_of_front = np.max(fitness[non_dominated, :], axis=0)
    extreme_points = get_extreme_points_c(
        fitness[non_dominated, :], ideal_point, extreme_points=extreme_points
    )
    nadir_point = get_nadir_point(
        extreme_points, ideal_point, worst_point, worst_of_population, worst_of_front
    )

    # Finding individuals in first 'n' fronts
    selection = np.asarray([], dtype=int)
    for front_id in range(len(fronts)):
        if len(np.concatenate(fronts[: front_id + 1])) < n_survive:
            continue
        else:
            fronts = fronts[: front_id + 1]
            selection = np.concatenate(fronts)
            break

    F = fitness[selection]

    last_front = fronts[-1]

    # Selecting individuals from the last acceptable front.
    if len(selection) > n_survive:
        niche_of_individuals, dist_to_niche = associate_to_niches(
            F, ref_dirs, ideal_point, nadir_point
        )
        # if there is only one front
        if len(fronts) == 1:
            n_remaining = n_survive
            until_last_front = np.array([], dtype=np.int)
            niche_count = np.zeros(len(ref_dirs), dtype=np.int)

        # if some individuals already survived
        else:
            until_last_front = np.concatenate(fronts[:-1])
            id_until_last_front = list(range(len(until_last_front)))
            niche_count = calc_niche_count(
                len(ref_dirs), niche_of_individuals[id_until_last_front]
            )
            n_remaining = n_survive - len(until_last_front)

        last_front_selection_id = list(range(len(until_last_front), len(selection)))
        if np.any(selection[last_front_selection_id] != last_front):
            print("error!!!")
        selected_from_last_front = niching(
            fitness[last_front, :],
            n_remaining,
            niche_count,
            niche_of_individuals[last_front_selection_id],
            dist_to_niche[last_front_selection_id],
        )
        final_selection = np.concatenate(
            (until_last_front, last_front[selected_from_last_front])
        )
        if extreme_points is None:
            print('Error')
        if final_selection is None:
            print('Error')
    else:
        final_selection = selection
    return(final_selection.astype(int), extreme_points)


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


def niching(F, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(F.shape[0], True)

    while len(survivors) < n_remaining:

        # all niches where new individuals can be assigned to
        next_niches_list = np.unique(niche_of_individuals[mask])

        # pick a niche with minimum assigned individuals - break tie if necessary
        next_niche_count = niche_count[next_niches_list]
        next_niche = np.where(next_niche_count == next_niche_count.min())[0]
        next_niche = next_niches_list[next_niche]
        next_niche = next_niche[np.random.randint(0, len(next_niche))]

        # indices of individuals that are considered and assign to next_niche
        next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

        # shuffle to break random tie (equal perp. dist) or select randomly
        np.random.shuffle(next_ind)

        if niche_count[next_niche] == 0:
            next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
        else:
            # already randomized through shuffling
            next_ind = next_ind[0]

        mask[next_ind] = False
        survivors.append(int(next_ind))

        niche_count[next_niche] += 1

    return survivors


def associate_to_niches(F, ref_dirs, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = calc_perpendicular_distance(N, ref_dirs)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


def calc_perpendicular_distance(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix
