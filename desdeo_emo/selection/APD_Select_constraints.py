from typing import Callable, List
from warnings import warn

import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import InteractiveDecompositionSelectionBase


class APD_Select(InteractiveDecompositionSelectionBase):
    """
    The selection operator for the RVEA algorithm. Read the following paper for more
    details.
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

    Parameters
    ----------
    pop : Population
        The population instance
    time_penalty_function : Callable
        A function that returns the time component in the penalty function.
    alpha : float, optional
        The RVEA alpha parameter, by default 2
    """

    def __init__(
        self,
        pop: Population,
        time_penalty_function: Callable,
        alpha: float = 2,
        selection_type: str = None,
    ):
        super().__init__(pop.pop_size, pop.problem.n_of_fitnesses, selection_type)
        self.time_penalty_function = time_penalty_function
        if alpha is None:
            alpha = 2
        self.alpha = alpha
        self.n_of_fitnesses = pop.problem.n_of_fitnesses

        self.ideal: np.ndarray = pop.ideal_fitness_val

    def do(self, pop: Population) -> List[int]:
        """Select individuals for mating on basis of Angle penalized distance.

        Parameters
        ----------
        pop : Population
            The current population.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        partial_penalty_factor = self._partial_penalty_factor()
        ref_vectors = self.vectors.neighbouring_angles_current
        # Normalization - There may be problems here
        fitness = self._calculate_fitness(pop)
        fmin = np.amin(fitness, axis=0)
        self.ideal = np.amin(
            np.vstack((self.ideal, fmin, pop.ideal_fitness_val)), axis=0
        )
        translated_fitness = fitness - self.ideal
        fitness_norm = np.linalg.norm(translated_fitness, axis=1)
        # TODO check if you need the next line
        # TODO changing the order of the following few operations might be efficient
        fitness_norm = np.repeat(fitness_norm, len(translated_fitness[0, :])).reshape(
            len(fitness), len(fitness[0, :])
        )
        # Convert zeros to eps to avoid divide by zero.
        # Has to be checked!
        fitness_norm[fitness_norm == 0] = np.finfo(float).eps
        normalized_fitness = np.divide(
            translated_fitness, fitness_norm
        )  # Checked, works.
        cosine = np.dot(normalized_fitness, np.transpose(self.vectors.values))
        if cosine[np.where(cosine > 1)].size:
            warn("RVEA.py line 60 cosine larger than 1 decreased to 1")
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            warn("RVEA.py line 64 cosine smaller than 0 increased to 0")
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector assignment
        assigned_vectors = np.argmax(cosine, axis=1)
        selection = np.array([], dtype=int)
        # Selection
        # Convert zeros to eps to avoid divide by zero.
        # Has to be checked!
        ref_vectors[ref_vectors == 0] = np.finfo(float).eps
        for i in range(0, len(self.vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )

            # Constraint check
            if len(sub_population_index) > 1 and pop.constraint is not None:
                violation_values = pop.constraint[sub_population_index]
                # violation_values = -violation_values
                violation_values = np.maximum(0, violation_values)
                # True if feasible
                feasible_bool = (violation_values == 0).all(axis=1)

                # Case when entire subpopulation is infeasible
                if (feasible_bool == False).all():
                    violation_values = violation_values.sum(axis=1)
                    sub_population_index = sub_population_index[
                        np.where(violation_values == violation_values.min())
                    ]
                # Case when only some are infeasible
                else:
                    sub_population_index = sub_population_index[feasible_bool]

            sub_population_fitness = translated_fitness[sub_population_index]
            # fast tracking singly selected individuals
            if len(sub_population_index) == 1:
                selx = sub_population_index
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
            elif len(sub_population_index) > 1:
                # APD Calculation
                angles = theta[sub_population_index, i]
                angles = np.divide(angles, ref_vectors[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                apd = np.multiply(
                    np.transpose(sub_pop_fitness_magnitude),
                    (1 + np.dot(partial_penalty_factor, angles)),
                )
                minidx = np.where(apd == np.nanmin(apd))
                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
        return selection.squeeze()

    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.
            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1

        Returns
        -------
        float
            The partial penalty value
        """
        penalty = self.time_penalty_function()
        if penalty < 0:
            penalty = 0
        if penalty > 1:
            penalty = 1
        penalty = (penalty ** self.alpha) * self.n_of_fitnesses
        return penalty
