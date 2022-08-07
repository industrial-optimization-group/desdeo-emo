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


class Prob_APD_select_v3_1(SelectionBase):
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
        # Normalization - There may be problems here
        #if ideal is not None:
        #    fmin = ideal
        #else:
        #    fmin = np.amin(fitness, axis=0)

        fmin = np.amin(fitness, axis=0)
        #fmin = np.array([0,0])
        translated_fitness = fitness - fmin
        pwrong = Probability_wrong(mean_values=translated_fitness, stddev_values=uncertainty, n_samples=1000)
        pwrong.vect_sample_f()


        fitness_norm = np.linalg.norm(pwrong.f_samples, axis=1)
        fitness_norm = np.repeat(np.reshape(fitness_norm, (len(fitness), 1, pwrong.n_samples)), len(fitness[0, :]), axis=1)

        normalized_fitness = np.divide(pwrong.f_samples, fitness_norm)  # Checked, works.


        #Find cosine angles for all the samples


        cosine = np.tensordot(normalized_fitness, np.transpose(vectors.values), axes=([1], [0]))
        cosine = np.transpose(cosine,(0,2,1))

        if cosine[np.where(cosine > 1)].size:
            #print(
            #    "RVEA.py line 60 cosine larger than 1 decreased to 1:"
            #)
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            #print(
            #    "RVEA.py line 64 cosine smaller than 0 decreased to 0:"
            #)
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector asub_population_indexssignment
        #pwrong.compute_pdf(cosine)
        # Compute rank of cos theta (to be vectorized)
        rank_cosine = np.mean(cosine,axis=2)
        #print("Rank cosine:")
        #print(rank_cosine)
        assigned_vectors = np.argmax(rank_cosine, axis=1)
        selection = np.array([], dtype=int)
        # Selection

        vector_selection = None


        for i in range(0, len(vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )
            sub_population_fitness = pwrong.f_samples[sub_population_index]
            #print(len(sub_population_fitness))

            if len(sub_population_fitness > 0):
                # APD Calculation
                angles = theta[sub_population_index, i]
                angles = np.divide(angles, refV[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                sub_popfm = np.reshape(sub_pop_fitness_magnitude, (1, len(sub_pop_fitness_magnitude[:,0]), pwrong.n_samples))
                #sub_popfm = np.transpose(sub_popfm, axes=)

                #sub_popfm = np.repeat(sub_popfm, pwrong.n_samples, axis=2)
                angles = np.reshape(angles,(1,len(angles),pwrong.n_samples))


                #### Overall Mean/Median of apd
                apd = np.multiply(
                    sub_popfm,
                    (1 + np.dot(penalty_factor, angles))
                    #np.repeat(np.reshape((1 + np.dot(penalty_factor, np.mean(angles, axis=2))), (1, len(sub_pop_fitness_magnitude[:,0]), 1)), 1000, axis=2),
                )
                #rank_apd = np.median(apd, axis=2)

                """
                ###### Actual probability computation with ECDF
                pwrong.pdf_list = {}
                pwrong.compute_pdf(apd)
                #pwrong.plt_density(apd)
                pwrong.compute_rank_vectorized2()
                rank_apd = pwrong.rank_prob_wrong
                #print(rank_apd)
                apd_elites = apd[0,np.where(rank_apd[0,:]<0),:]
                if np.size(apd_elites) >= 1:
                    rank_apd = np.mean(apd_elites, axis=2)
                """

                ######## Mean based ranking for superfast computing
                rank_apd = np.mean(apd, axis=2)
                minidx = np.where(rank_apd[0] == np.nanmin(rank_apd[0]))

                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                #print(selx)
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                    vector_selection = np.asarray(i)
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
                    vector_selection = np.hstack((vector_selection, i))

        if selection.shape[0] == 1:
            print("Only one individual!!")
            rand_select = np.random.randint(len(fitness), size=1)
            #if rand_select == selection[0,0]:
            #    rand_select = np.random.randint(len(fitness), size=1)
            selection = np.vstack((selection, np.transpose(rand_select[0])))

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
        penalty = ((self.time_penalty_function()) ** self.alpha) * self.n_of_objectives
        if penalty < 0:
            penalty = 0
        if penalty > 1:
            penalty = 1
        return penalty