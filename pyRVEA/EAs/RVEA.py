"""The python version reference vector guided evolutionary algorithm.

See the details of RVEA in the following paper

R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
A Reference Vector Guided Evolutionary Algorithm for Many-objective
Optimization, IEEE Transactions on Evolutionary Computation, 2016

The source code of cRVEA is implemented by Bhupinder Saini

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi
Project researcher at University of Jyväskylä.
"""


from typing import TYPE_CHECKING

from pyRVEA.Selection.APD_select import APD_select

if TYPE_CHECKING:
    from pyRVEA.Population.Population import Population
    from pyRVEA.Problem.baseProblem import baseProblem
    from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors
    from pyRVEA.allclasses import Parameters


def rvea(
    population: "Population",
    problem: "baseProblem",
    parameters: "Parameters",
    reference_vectors: "ReferenceVectors",
    progressbar: "tqdm",
):
    """
    Run RVEA.

    This only conducts reproduction and selection. Reference vector adaptation should
    be done outside. Changes variable population.

    Parameters
    ----------
    population : Population
        This variable is updated as evolution takes place
    problem : Problem
        Contains the details of the problem.
    parameters : Parameters
        Contains the hyper-parameters of RVEA evolution.
    reference_vectors : ReferenceVectors
        Class containing the reference vectors.
    progressbar : tqdm or tqdm_notebook
        An iterable used to display the progress bar.

    Returns
    -------
    Population
        Returns the Population after evolution.

    """
    refV = reference_vectors.neighbouring_angles()
    progress = progressbar(
        range(parameters["generations"]), desc="Generations", leave=False
    )
    for gen_count in progress:
        offspring = population.mate()
        population.add(offspring, problem)
        # APD Based selection
        penalty_factor = (
            (gen_count / parameters["generations"]) ** parameters["Alpha"]
        ) * problem.num_of_objectives
        select = APD_select(population.fitness, reference_vectors, penalty_factor, refV)
        population.keep(select)
    progress.close()
    return population
