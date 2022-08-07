import numpy as np
from warnings import warn
from typing import List, Callable
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from typing import TYPE_CHECKING
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong
import os
os.environ["OMP_NUM_THREADS"] = "1"

if TYPE_CHECKING:
    from pyRVEA.allclasses import ReferenceVectors


class Prob_Hybrid_APD_Select_v3(SelectionBase):
    """The selection operator for the RVEA algorithm. Read the following paper for more
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
        self, pop: Population, time_penalty_function: Callable, alpha: float = 2
    ):
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.n_of_objectives = pop.problem.n_of_objectives

    def do(self, pop: Population, vectors: ReferenceVectors) -> List[int]:
        """Select individuals for mating on basis of APD and probabilistic APD.

        Args:
            fitness (list): Fitness of the current population.

            uncertainty (list) : Uncertainty of the predicted objective values

            vectors (ReferenceVectors): Class containing reference vectors.

            penalty_factor (float): Multiplier of angular deviation from Reference
                vectors. See RVEA paper for details.

            ideal (list): ideal point for the population.
                Uses the min fitness value if None.

        Returns:
            [type]: A list of indices of the selected individuals.
        """
        fitness = pop.fitness
        uncertainty = pop.uncertainity
        penalty_factor = self._partial_penalty_factor()
        refV = vectors.neighbouring_angles_current
        fmin = np.amin(fitness, axis=0)
        translated_fitness = fitness - fmin
        ##### Generic sub-population
        # Normalization - There may be problems here

        fitness_norm_gen = np.linalg.norm(translated_fitness, axis=1)
        fitness_norm_gen = np.repeat(fitness_norm_gen, len(translated_fitness[0, :])).reshape(
            len(fitness), len(fitness[0, :])
        )
        normalized_fitness_gen = np.divide(translated_fitness, fitness_norm_gen)  # Checked, works.
        cosine_gen = np.dot(normalized_fitness_gen, np.transpose(vectors.values))
        if cosine_gen[np.where(cosine_gen > 1)].size:
            cosine_gen[np.where(cosine_gen > 1)] = 1
        if cosine_gen[np.where(cosine_gen < 0)].size:
            cosine_gen[np.where(cosine_gen < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta_gen = np.arccos(cosine_gen)
        # Reference vector asub_population_indexssignment
        assigned_vectors_gen = np.argmax(cosine_gen, axis=1)
        selection_gen = np.array([], dtype=int)

        #########################################################
        ##### Prob sub-population

        pwrong = Probability_wrong(mean_values=translated_fitness, stddev_values=uncertainty, n_samples=1000)
        pwrong.vect_sample_f()
        fitness_norm_prob = np.linalg.norm(pwrong.f_samples, axis=1)
        fitness_norm_prob = np.repeat(np.reshape(fitness_norm_prob, (len(fitness), 1, pwrong.n_samples)), len(fitness[0, :]), axis=1)
        normalized_fitness_prob = np.divide(pwrong.f_samples, fitness_norm_prob)  # Checked, works.

        # Find cosine angles for all the samples
        cosine_prob = np.tensordot(normalized_fitness_prob, np.transpose(vectors.values), axes=([1], [0]))
        cosine_prob = np.transpose(cosine_prob, (0, 2, 1))

        if cosine_prob[np.where(cosine_prob > 1)].size:
            cosine_prob[np.where(cosine_prob > 1)] = 1
        if cosine_prob[np.where(cosine_prob < 0)].size:
            cosine_prob[np.where(cosine_prob < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta_prob = np.arccos(cosine_prob)
        # Compute rank of cos theta (to be vectorized)
        rank_cosine_prob = np.mean(cosine_prob, axis=2)
        assigned_vectors_prob = np.argmax(rank_cosine_prob, axis=1)
        selection_prob = np.array([], dtype=int)

        selection = np.array([], dtype=int)
        #print("**************New Generation************")
        # Selection
        for i in range(0, len(vectors.values)):
            #print("sub pop index:",i)
            #sub_population_index_prob = np.atleast_1d(
            #    np.squeeze(np.where(assigned_vectors_prob == i))
            #)
            sub_population_index_prob = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors_gen == i))
            )
            sub_population_index_gen = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors_gen == i))
            )
            sub_population_fitness_prob = pwrong.f_samples[sub_population_index_prob]
            #sub_population_fitness_prob = pwrong.f_samples[sub_population_index_gen]
            sub_population_fitness_gen = translated_fitness[sub_population_index_gen]
            
            if len(sub_population_fitness_gen) > 0:
                ############ Generic APD ################     

                # APD Calculation
                angles_gen = theta_gen[sub_population_index_gen, i]
                angles_gen = np.divide(angles_gen, refV[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude_gen = np.sqrt(
                    np.sum(np.power(sub_population_fitness_gen, 2), axis=1)
                )
                apd_gen = np.multiply(
                    np.transpose(sub_pop_fitness_magnitude_gen),
                    (1 + np.dot(penalty_factor, angles_gen)),
                )
                minidx_gen = np.where(apd_gen == np.nanmin(apd_gen))
                #print("Id gen:",minidx_gen)
                if np.isnan(apd_gen).all():
                    continue
                selx_gen = sub_population_index_gen[minidx_gen]
                #print("Selection gen:",selx_gen)
                if selection_gen.shape[0] == 0:
                    selection_gen = np.hstack((selection_gen, np.transpose(selx_gen[0])))
                else:
                    selection_gen = np.vstack((selection_gen, np.transpose(selx_gen[0])))    

            if len(sub_population_fitness_prob) > 0:
                
                angles_prob = theta_prob[sub_population_index_prob, i]
                angles_prob = np.divide(angles_prob, refV[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude_prob = np.sqrt(
                    np.sum(np.power(sub_population_fitness_prob, 2), axis=1)
                )
                sub_popfm_prob = np.reshape(sub_pop_fitness_magnitude_prob,
                                    (1, len(sub_pop_fitness_magnitude_prob[:, 0]), pwrong.n_samples))
                angles_prob = np.reshape(angles_prob, (1, len(angles_prob), pwrong.n_samples))

                apd_prob = np.multiply(
                    sub_popfm_prob,
                    (1 + np.dot(penalty_factor, angles_prob)))
                
                # Using mean APD
                rank_apd_prob = np.mean(apd_prob, axis=2)


                minidx_prob = np.where(rank_apd_prob[0] == np.nanmin(rank_apd_prob[0]))
                #print("Id prob:",minidx_prob)
                if np.isnan(apd_prob).all():
                    continue
                selx_prob = sub_population_index_prob[minidx_prob]
                #print("Selection prob:",selx_prob)
                if selection_prob.shape[0] == 0:
                    selection_prob = np.hstack((selection_prob, np.transpose(selx_prob[0])))
                else:
                    selection_prob = np.vstack((selection_prob, np.transpose(selx_prob[0])))                            
        selection_gen = selection_gen.squeeze()
        selection_prob = selection_prob.squeeze()
        #print("Selection gen:",selection_gen)
        #print("Selection prob:",selection_prob)
        selection = np.union1d(selection_gen,selection_prob)
        #print("Selection union:",selection)
        while selection.shape[0] == 1:
            rand_select = np.random.randint(len(fitness), size=1)
            #selection = np.vstack((selection, np.transpose(rand_select[0])))
            selection = np.union1d(selection,np.transpose(rand_select[0]))
        selection = selection.squeeze()
        #print("Selection all:",selection)
        return selection

    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.
            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1

        Returns
        -------
        float
            The partial penalty value
        """
        if self.time_penalty_function() < 0:
            px = 0
        elif self.time_penalty_function() > 1:
            px = 1
        else:
            px= self.time_penalty_function()
        penalty = ((px) ** self.alpha) * self.n_of_objectives

        return penalty