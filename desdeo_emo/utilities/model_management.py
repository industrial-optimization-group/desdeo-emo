from numpy.core.numeric import indices
from numpy.lib.arraysetops import unique
from desdeo_tools.scalarization.ASF import SimpleASF, ReferencePointASF
from numba import njit
import numpy as np
import pandas as pd
import copy
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,\
    WhiteKernel, RBF, Matern, ConstantKernel


#TODO: add @njit function here
def remove_duplicate(X, archive_x):
    """identifiesthe duplicate rows for decision variables
    Args:
    X (np.ndarray): the current decision variables.
    archive_x (np.ndarray): The decision variables in the archive.

    Returns: 
    indicies (np.ndarray): the indicies of solutions that are NOT already in the archive.
    """
    indicies = None
    archive = archive_x.to_numpy()
    tmp = X
    for i in archive:
        for k in range(len(X)):
            for j in range(len(i)):
                tmp[k,j] = X[k,j] - i[j]
        tmp = np.round(tmp, 2)
        if indicies is None:
            indicies = np.where(~tmp.any(axis=1))[0]
        else:
            tmp = np.where(~tmp.any(axis=1))[0]
            if tmp.size > 0:
                indicies = np.hstack((indicies.squeeze(),tmp))
        if indicies.size == 0:
            indicies = None
    if indicies is None:
        return None
    else:

        return indicies


def ikrvea_mm(reference_point, evolver , problem, u: int) -> float:
    """ Selects the solutions that need to be reevaluated with the original functions.
    This model management is based on the following papaer: 

    'P. Aghaei Pour, T. Rodemann, J. Hakanen, and K. Miettinen, “Surrogate assisted interactive
     multiobjective optimization in energy system design of buildings,” 
     Optimization and Engineering, 2021.'

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        evolver : the object that contains population class.
        problem : the problem class

    Returns:
        float: the new problem object that has an updated archive.
    """
    archive = problem.archive.to_numpy()
    surrogate_obj = copy.deepcopy( evolver.population.objectives)
    decision_variables = copy.deepcopy(evolver.population.individuals)
    unc = copy.deepcopy(evolver.population.uncertainity)
    #pd.concat([b,b], ignore_index= True)
    nd = remove_duplicate(decision_variables, problem.archive.drop(
            problem.objective_names, axis=1)) #removing duplicate solutions
    if nd is not None:
        non_duplicate_dv = evolver.population.individuals[nd]
        non_duplicate_obj = evolver.population.objectives[nd]
        non_duplicate_unc = evolver.population.uncertainity[nd]
    else:
        non_duplicate_dv = evolver.population.individuals
        non_duplicate_obj = evolver.population.objectives
        non_duplicate_unc = evolver.population.uncertainity

    # Selecting solutions with lowest ASF values
    asf_solutions = SimpleASF([1]*problem.n_of_objectives).__call__(non_duplicate_obj, reference_point)
    idx = np.argpartition(asf_solutions, 2*u)
    asf_unc = np.max(non_duplicate_unc [idx[0:2*u]], axis= 1)
    # index of solutions with lowest Uncertainty
    lowest_unc_index = np.argpartition(asf_unc, u)[0:u]
    names = np.hstack((problem.variable_names,problem.objective_names))
    reevaluated_objs = problem.evaluate(non_duplicate_dv[lowest_unc_index], use_surrogate=False)[0]
    new_results = np.hstack((non_duplicate_dv[lowest_unc_index], reevaluated_objs))
    archive = np.vstack((archive, new_results))
    new_archive = pd.DataFrame(archive, columns=names)
    problem.archive = new_archive #updating the archive
    problem.train(models=GaussianProcessRegressor,\
         model_parameters={'kernel': Matern(nu=1.5)}) 

    return problem


