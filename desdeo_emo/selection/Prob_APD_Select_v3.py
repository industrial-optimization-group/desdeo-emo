import numpy as np
from warnings import warn
from typing import List, Callable

from numpy.core.fromnumeric import size
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from typing import TYPE_CHECKING
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong
import os
import matplotlib.pyplot as plt
from matplotlib import rc

os.environ["OMP_NUM_THREADS"] = "1"



class Prob_APD_select_v3(SelectionBase):
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
        
        fitness = pop.fitness
        uncertainty = pop.uncertainity
        penalty_factor = self._partial_penalty_factor()
        """
        penalty_factor = 2.0

        fitness = np.array([[1.2, 0.6],
                            [1.23, 0.65],
                            [0.2, 1.5],
                            [0.2, 1.5],
                            [0.6,1.2],
                            [0.7, 1],
                            [1.4, 0.1],
                            [1.38, 0.09]])

        uncertainty = np.array([[0.1, 0.2],
                                [0.03, 0.05],
                                [0.1, 0.1],
                                [0.05, 0.05],
                                [0.05,0.1],
                                [0.1,0.05],
                                [0.1, 0.05],
                                [0.1, 0.05]])

        """

        #def Prob_APD_select_v3(
        #    fitness: list,
        #    uncertainty: list,
        #    vectors: "ReferenceVectors",
        #    penalty_factor: float,
        #    ideal: list = None,
        #):

        
        refV = vectors.neighbouring_angles_current
        fmin = np.amin(fitness, axis=0)
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
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector asub_population_indexssignment
        #pwrong.compute_pdf(cosine)
        # Compute rank of cos theta (to be vectorized)
        rank_cosine = np.mean(cosine,axis=2)
        assigned_vectors = np.argmax(rank_cosine, axis=1)
        selection = np.array([], dtype=int)

        vector_selection = None

        for i in range(0, len(vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )
            sub_population_fitness = pwrong.f_samples[sub_population_index]

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
                angles = np.reshape(angles,(1,len(angles),pwrong.n_samples))


                #### Overall Mean/Median of apd
                apd = np.multiply(
                    sub_popfm,
                    (1 + np.dot(penalty_factor, angles))
                )
                #rank_apd = np.mean(apd, axis=2)
                rank_apd = pwrong.compute_rank_MC(apd)
                minidx = np.where(rank_apd[0] == np.nanmin(rank_apd[0]))

                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                    vector_selection = np.asarray(i)
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
                    vector_selection = np.hstack((vector_selection, i))

        if selection.shape[0] == 1:
            print("Only one individual!!")
            rand_select = np.random.randint(len(fitness), size=1)
            selection = np.vstack((selection, np.transpose(rand_select[0])))
        #print("Selection:",selection)
        assigned_vectors = np.argmax(rank_cosine, axis=1)
        #plot_selection(selection, fitness, uncertainty, assigned_vectors, vectors)

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
        if self.time_penalty_function() < 0:
            px = 0
        elif self.time_penalty_function() > 1:
            px = 1
        else:
            px= self.time_penalty_function()
        penalty = ((px) ** self.alpha) * self.n_of_objectives

        return penalty


def plot_selection(selection, fitness, uncertainty, assigned_vectors, vectors):
    rc('font',**{'family':'serif','serif':['Helvetica']})
    rc('text', usetex=True)
    plt.rcParams.update({'font.size': 17})
    #plt.rcParams["text.usetex"] = True
    vector_anno = np.arange(len(vectors.values))
    fig = plt.figure(1, figsize=(10, 10))
    fig.set_size_inches(4, 4)
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    plt.xlim(-0.02, 1.75)
    plt.ylim(-0.02, 1.75)

    #plt.scatter(vectors.values[:, 0], vectors.values[:, 1])
    sx= selection.squeeze()
    #plt.scatter(translated_fitness[sx,0],translated_fitness[sx,1])
    #plt.scatter(fitness[sx, 0], fitness[sx, 1])
    #for i, txt in enumerate(vector_anno):
    #    ax.annotate(vector_anno, (vectors.values[i, 0], vectors.values[i, 1]))
    
    [plt.arrow(0, 0, dx, dy, color='magenta', length_includes_head=True,
               head_width=0.02, head_length=0.04) for ((dx, dy)) in vectors.values]

    plt.errorbar(fitness[:, 0], fitness[:, 1], xerr=1.96 * uncertainty[:, 0], yerr=1.96 * uncertainty[:, 1],fmt='*', ecolor='r', c='r')
    for i in vector_anno:
        ax.annotate(vector_anno[i], ((vectors.values[i, 0]+0.01), (vectors.values[i, 1]+0.01)))


    plt.errorbar(fitness[sx,0],fitness[sx,1], xerr=1.96*uncertainty[sx, 0], yerr=1.96*uncertainty[sx, 1],fmt='o', ecolor='g',c='g')
    for i in range(len(assigned_vectors)):
        if np.isin(i, sx):
            ax.annotate(assigned_vectors[i], ((fitness[i, 0]-0.1), (fitness[i, 1]-0.1)))
        else:
            ax.annotate(assigned_vectors[i], ((fitness[i, 0]+0.01), (fitness[i, 1]+0.01)))
    plt.show()
    fig.savefig('t_select.pdf',bbox_inches='tight')
    ax.cla()
    fig.clf()
    print("plotted!")
    
    ######### for distribution plots
    """
    fig = plt.figure(1, figsize=(6, 6))
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(111)
    plt.xlim(-0.02, 1.1)
    plt.ylim(-0.02, 1.1)
    vector_anno = np.arange(len(vectors.values))
    viridis = cm.get_cmap('viridis',len(vectors.values[:,0]))
    samples = pwrong.f_samples.tolist()
    samplesx = samples[0][0]
    samplesy = samples[0][1]
    #samplesx = samples[0][0] + samples[1][0]
    #samplesy = samples[0][1] + samples[1][1]
    colour = np.argmax(cosine, axis=1)
    coloursamples = colour.tolist()[0]
    #coloursamples = colour.tolist()[0] + colour.tolist()[1]

    colourvectors = list(range(0,len(vectors.values[:,0])))
    #colourvectors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    [plt.arrow(0,0,dx, dy, color=viridis.colors[c],length_includes_head=True,
          head_width=0.02, head_length=0.04) for ((dx,dy),c) in zip(vectors.values, colourvectors)]
    for i in range(len(assigned_vectors)-1):
        ax.annotate(assigned_vectors[i], ((translated_fitness[i, 0]+0.08), (translated_fitness[i, 1]+0.03)))
    for i in vector_anno:
        ax.annotate(vector_anno[i], ((vectors.values[i, 0]+0.01), (vectors.values[i, 1]+0.01)))

    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    plt.scatter(samplesx, samplesy, c=viridis.colors[coloursamples], s=0.2)
    plt.scatter(translated_fitness[0][0], translated_fitness[0][1], c='r', s=10, marker='*')
    ellipse = Ellipse(xy=(translated_fitness[0][0], translated_fitness[0][1]), width=uncertainty[0][0] * 1.96 * 2,
                      height=uncertainty[0][1] * 1.96 * 2,
                      edgecolor='r', fc='None', lw=1)
    ax.add_patch(ellipse)
    plt.show()
    print((colour[0, :] == 3).sum())
    print((colour[0, :] == 2).sum())
    #print(selection)
    #fig.savefig('t1.pdf')
    """