from desdeo_emo.selection.NSGAIII_select import NSGAIII_select
import numpy as np
#from pygmo import fast_non_dominated_sorting as nds
from desdeo_tools.utilities import fast_non_dominated_sort
from typing import List
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors


class IOPIS_NSGAIII_select(NSGAIII_select):
    def __init__(
        self,
        scalarization_nethods,
        pop: Population,
        n_survive: int = None,
        selection_type: str = None,
    ):
        self.scalarization_methods = scalarization_nethods
        self.worst_fitness: np.ndarray = -np.full(
            (1, len(scalarization_nethods)), np.inf
        )
        self.extreme_points: np.ndarray = None
        if n_survive is None:
            self.n_survive: int = pop.pop_size
        self.ideal: np.ndarray = np.full((1, len(scalarization_nethods)), np.inf)

    def do(
        self, pop: Population, vectors: ReferenceVectors, reference_point: np.ndarray
    ) -> List[int]:
        """Select individuals for mating for NSGA-III.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        ref_dirs = vectors.values_planar
        fitness = np.asarray(
            [
                scalar(pop.fitness, reference_point)
                for scalar in self.scalarization_methods
            ]
        ).T
        # Calculating fronts and ranks
        # fronts, dl, dc, rank = nds(fitness)
        fronts = fast_non_dominated_sort(fitness)
        fronts = [np.where(fronts[i])[0] for i in range(len(fronts))]
        non_dominated = fronts[0]
        fmin = np.amin(fitness, axis=0)
        self.ideal = np.amin(np.vstack((self.ideal, fmin)), axis=0)

        # Calculating worst points
        self.worst_fitness = np.amax(np.vstack((self.worst_fitness, fitness)), axis=0)
        worst_of_population = np.amax(fitness, axis=0)
        worst_of_front = np.max(fitness[non_dominated, :], axis=0)
        self.extreme_points = self.get_extreme_points_c(
            fitness[non_dominated, :], self.ideal, extreme_points=self.extreme_points
        )
        nadir_point = self.get_nadir_point(
            self.extreme_points,
            self.ideal,
            self.worst_fitness,
            worst_of_population,
            worst_of_front,
        )

        # Finding individuals in first 'n' fronts
        selection = np.asarray([], dtype=int)
        for front_id in range(len(fronts)):
            if len(np.concatenate(fronts[: front_id + 1])) < self.n_survive:
                continue
            else:
                fronts = fronts[: front_id + 1]
                selection = np.concatenate(fronts)
                break

        F = fitness[selection]

        last_front = fronts[-1]

        # Selecting individuals from the last acceptable front.
        if len(selection) > self.n_survive:
            niche_of_individuals, dist_to_niche = self.associate_to_niches(
                F, ref_dirs, self.ideal, nadir_point
            )
            # if there is only one front
            if len(fronts) == 1:
                n_remaining = self.n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                id_until_last_front = list(range(len(until_last_front)))
                niche_count = self.calc_niche_count(
                    len(ref_dirs), niche_of_individuals[id_until_last_front]
                )
                n_remaining = self.n_survive - len(until_last_front)

            last_front_selection_id = list(range(len(until_last_front), len(selection)))
            if np.any(selection[last_front_selection_id] != last_front):
                print("error!!!")
            selected_from_last_front = self.niching(
                fitness[last_front, :],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front_selection_id],
                dist_to_niche[last_front_selection_id],
            )
            final_selection = np.concatenate(
                (until_last_front, last_front[selected_from_last_front])
            )
            if self.extreme_points is None:
                print("Error")
            if final_selection is None:
                print("Error")
        else:
            final_selection = selection
        return final_selection.astype(int)

