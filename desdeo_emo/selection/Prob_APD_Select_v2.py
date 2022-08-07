import numpy as np
from warnings import warn
from typing import TYPE_CHECKING
from desdeo_emo.othertools.Probabilitywrong import Probability_wrong
import matplotlib.pyplot as plt
from kneed import KneeLocator
# from pygmo import fast_non_dominated_sorting as nds
from pygmo import non_dominated_front_2d as nd2
from matplotlib import cm
from matplotlib.patches import Ellipse
import os
import multiprocessing as mp
import itertools
from itertools import repeat
from joblib import Parallel, delayed

#os.environ["OMP_NUM_THREADS"] = "1"

if TYPE_CHECKING:
    from pyRVEA.allclasses import ReferenceVectors


def select_apd_2(pwrong, assigned_vectors, theta, refV, penalty_factor, vectors
               ):
    apd_list = {}
    indiv_index_list = {}
    for i in range(len(vectors.values)):
        sub_population_index = np.atleast_1d(
            np.squeeze(np.where(assigned_vectors == i))
        )
        sub_population_fitness = pwrong.f_samples[sub_population_index]
        #print(len(sub_population_fitness))
        if len(sub_population_fitness) > 0:
            # APD Calculation
            angles = theta[sub_population_index, i]
            angles = np.divide(angles, refV[i])  # This is correct.
            # You have done this calculation before. Check with fitness_norm
            # Remove this horrible line
            sub_pop_fitness_magnitude = np.sqrt(
                np.sum(np.power(sub_population_fitness, 2), axis=1)
            )
            sub_popfm = np.reshape(sub_pop_fitness_magnitude,
                                   (1, len(sub_pop_fitness_magnitude[:, 0]), pwrong.n_samples))
            angles = np.reshape(angles, (1, len(angles), pwrong.n_samples))

            #### Overall Mean/Median of apd
            apd = np.multiply(
                sub_popfm,
                (1 + np.dot(penalty_factor, angles))

            )

            apd_list[str(i)] = apd
            indiv_index_list[str(i)] = sub_population_index

            #pwrong.compute_pdf(apd)
            #pwrong.plt_density(apd)
    return apd_list, indiv_index_list



def select_apd(i, pwrong, assigned_vectors, theta, refV, penalty_factor
               ):

    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == i))
    )
    sub_population_fitness = pwrong.f_samples[sub_population_index]
    #print(len(sub_population_fitness))
    if len(sub_population_fitness) == 1:
        selx = sub_population_index[0]
        return selx

    elif len(sub_population_fitness) > 0:
        # APD Calculation
        angles = theta[sub_population_index, i]
        angles = np.divide(angles, refV[i])  # This is correct.
        # You have done this calculation before. Check with fitness_norm
        # Remove this horrible line
        sub_pop_fitness_magnitude = np.sqrt(
            np.sum(np.power(sub_population_fitness, 2), axis=1)
        )
        sub_popfm = np.reshape(sub_pop_fitness_magnitude,
                               (1, len(sub_pop_fitness_magnitude[:, 0]), pwrong.n_samples))
        # sub_popfm = np.transpose(sub_popfm, axes=)

        # sub_popfm = np.repeat(sub_popfm, pwrong.n_samples, axis=2)
        angles = np.reshape(angles, (1, len(angles), pwrong.n_samples))

        #### Overall Mean/Median of apd
        apd = np.multiply(
            sub_popfm,
            (1 + np.dot(penalty_factor, angles))
            # np.repeat(np.reshape((1 + np.dot(penalty_factor, np.mean(angles, axis=2))), (1, len(sub_pop_fitness_magnitude[:,0]), 1)), 1000, axis=2),
        )
        # rank_apd = np.median(apd, axis=2)

        ###### Actual probability computation with ECDF
        #print("Computing Probabilitites!")
        pwrong.pdf_list = {}
        pwrong.compute_pdf(apd)
        #pwrong.plt_density(apd)
        pwrong.compute_rank_vectorized2()
        rank_apd = pwrong.rank_prob_wrong
        print("Subpopulation:")
        print(i)
        print("Ranking:")
        print(rank_apd)
        apd_elites = apd[0, np.where(rank_apd[0, :] < 0), :]
        if np.size(apd_elites) >= 1:
            rank_apd = np.mean(apd_elites, axis=2)


        minidx = np.where(rank_apd[0] == np.nanmin(rank_apd[0]))

        if np.isnan(apd).all():
            return -1
        selx = sub_population_index[minidx]
        # print(selx)
        return np.transpose(selx[0])


def fun_wrapper(indices):
    return select_apd(*indices)

def Prob_APD_select_v2(
        fitness: list,
        uncertainty: list,
        vectors: "ReferenceVectors",
        penalty_factor: float,
        ideal: list = None,
):
    """Select individuals for mating on basis of Angle penalized distance.

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


    penalty_factor = 1.0
    fitness = np.array([[0.6, 0.6],
                       [0.1, 0.1],
                        ])
    uncertainty = np.array([[0.03, 0.1],
                            [0.1, 0.1],
                            ])
    #uncertainty = np.array([[0.1, 0.03],
    #                        [0.1, 0.1],
    #                        ])
    """
    #penalty_factor = 1.0
    # fitness = np.array([[1, 1],[1.2, 1.2]])
    # uncertainty = np.array([[0.8, 0.09],[0.09, 0.8]])
    # penalty_factor = 2.0
    # print(penalty_factor)
    refV = vectors.neighbouring_angles_current
    # Normalization - There may be problems here
    # if ideal is not None:
    #    fmin = ideal
    # else:
    #    fmin = np.amin(fitness, axis=0)

    fmin = np.amin(fitness, axis=0)
    # fmin = np.array([0,0])
    translated_fitness = fitness - fmin
    pwrong = Probability_wrong(mean_values=translated_fitness, stddev_values=uncertainty, n_samples=1000)
    pwrong.vect_sample_f()
    # pwrong.f_samples = pwrong.f_samples - np.reshape(fmin,(1,2,1))

    """
    pwrong = Probability_wrong(mean_values=fitness, stddev_values=uncertainty, n_samples=1000)
    pwrong.vect_sample_f()
    fmin = np.amin(pwrong.f_samples, axis=(0,2))
    translated_fitness = fitness - fmin
    pwrong.f_samples = pwrong.f_samples - np.reshape(fmin,(1,2,1))
    """

    fitness_norm = np.linalg.norm(pwrong.f_samples, axis=1)
    fitness_norm = np.repeat(np.reshape(fitness_norm, (len(fitness), 1, pwrong.n_samples)), len(fitness[0, :]), axis=1)

    normalized_fitness = np.divide(pwrong.f_samples, fitness_norm)  # Checked, works.

    # Find cosine angles for all the samples

    """
    cosine = None
    for i in range(pwrong.n_samples):
        cosine_temp = np.dot(normalized_fitness[:,:,i], np.transpose(vectors.values))
        cosine_temp = np.reshape(cosine_temp,(len(fitness), len(vectors.values), 1))
        if cosine is None:
            cosine = cosine_temp
        else:
            np.shape(cosine_temp)
            np.shape(cosine)
            cosine = np.concatenate((cosine, cosine_temp),axis=2)
    """

    cosine = np.tensordot(normalized_fitness, np.transpose(vectors.values), axes=([1], [0]))
    cosine = np.transpose(cosine, (0, 2, 1))

    if cosine[np.where(cosine > 1)].size:
        # print(
        #    "RVEA.py line 60 cosine larger than 1 decreased to 1:"
        # )
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        # print(
        #    "RVEA.py line 64 cosine smaller than 0 decreased to 0:"
        # )
        cosine[np.where(cosine < 0)] = 0
    # Calculation of angles between reference vectors and solutions
    theta = np.arccos(cosine)
    # Reference vector asub_population_indexssignment
    # pwrong.compute_pdf(cosine)
    # Compute rank of cos theta (to be vectorized)
    rank_cosine = np.mean(cosine, axis=2)
    # print("Rank cosine:")
    # print(rank_cosine)
    assigned_vectors = np.argmax(rank_cosine, axis=1)
    selection = np.array([], dtype=int)
    # Selection

    vector_selection = None

    # fig = plt.figure(1, figsize=(6, 6))
    # ax = fig.add_subplot(111)

    #for i in range(0, len(vectors.values)):





    #input = ((i) for i in
    #         itertools.product(range(len(vectors.values))))

    #input = ((i, j, k) for i, j, k in
    #         itertools.product(range(len(vectors.values)),pwrong, assigned_vectors, theta, refV, penalty_factor))

    #results = p.map(select_apd, (input, pwrong, assigned_vectors, theta, refV, penalty_factor))

    #p = mp.Pool(int(mp.cpu_count()))
    #results = p.map(fun_wrapper, (
    #    range(len(vectors.values)), repeat(pwrong), repeat(assigned_vectors), repeat(theta), repeat(refV), repeat(penalty_factor)))
    #p.close()
    #p.join()


    ############## Parallel Process 1
    """
    worker = Parallel(n_jobs=20,pre_dispatch='all')

    res=worker(delayed(select_apd)(i,pwrong, assigned_vectors, theta, refV, penalty_factor) for i in range(len(vectors.values)))
    print(res)
    results = []
    for val in res:
        if val != None:
            results.append(val)
    selection = np.asarray(results)
    #selection = np.reshape(results, (0,len(results)))
    #selection = [x for x in selection if x != -1]
    """
    apd_list, inidiv_index_list = select_apd_2(pwrong, assigned_vectors, theta, refV, penalty_factor, vectors)
    selection = pwrong.compute_rank_vectorized_apd(apd_list, inidiv_index_list)

    if selection.shape[0] == 1:
        print("Only one individual!!")
        rand_select = np.random.randint(len(fitness), size=1)
        # if rand_select == selection[0,0]:
        #    rand_select = np.random.randint(len(fitness), size=1)
        selection = np.vstack((selection, np.transpose(rand_select[0])))

    # Plots here

    """
    #plt.rcParams["text.usetex"] = True
    vector_anno = np.arange(len(vectors.values))
    fig = plt.figure(1, figsize=(6, 6))
    fig.set_size_inches(5, 5)
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

    [plt.arrow(0, 0, dx, dy, color='b', length_includes_head=True,
               head_width=0.02, head_length=0.04) for ((dx, dy)) in vectors.values]

    plt.errorbar(fitness[:, 0], fitness[:, 1], xerr=1.96 * uncertainty[:, 0], yerr=1.96 * uncertainty[:, 1],fmt='o', ecolor='r', c='r')
    for i in vector_anno:
        ax.annotate(vector_anno[i], ((vectors.values[i, 0]+0.01), (vectors.values[i, 1]+0.01)))


    plt.errorbar(fitness[sx,0],fitness[sx,1], xerr=1.96*uncertainty[sx, 0], yerr=1.96*uncertainty[sx, 1],fmt='o', ecolor='g',c='g')
    for i in range(len(assigned_vectors)):
        ax.annotate(assigned_vectors[i], ((fitness[i, 0]+0.01), (fitness[i, 1]+0.01)))
        #ax.annotate(assigned_vectors[i], ((fitness[i, 0] + 0.01), (fitness[i, 1] + 0.01)))
    plt.show()
    #fig.savefig('t_select.pdf')

    ######### for distribution plots

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
    return selection.squeeze()
