"""Testing code."""

from collections import Sequence
from itertools import combinations
from random import shuffle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optproblems import dtlz, zdt
from pandas.plotting import parallel_coordinates
from pyDOE import lhs
from pygmo import fast_non_dominated_sorting as nds
from pygmo import hypervolume as hv
from pygmo import non_dominated_front_2d as nd2
from scipy.special import comb
from tqdm import tqdm, tqdm_notebook

from pyRVEA.RVEA import rvea


class Problem:
    """Base class for the problems."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """Pydocstring is ruthless."""
        self.name = name
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.num_of_constraints = num_of_constraints
        self.obj_func = []
        self.upper_limits = upper_limits
        self.lower_limits = lower_limits

    def objectives(self, decision_variables):
        """
        Accept a sample.

        Return all corresponding objective values as a
        list or array.
        """
        pass

    def constraints():
        """
        Accept a sample and/or corresponding objective values.

        Return all corresponding constraint violation values as a
        list or array.
        """
        pass

    def update():
        """Update the problem based on new information."""
        pass


class testProblem(Problem):
    """Defines the problem."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """Pydocstring is ruthless."""
        super(testProblem, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )
        if name == "ZDT1":
            self.obj_func = zdt.ZDT1()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT2":
            self.obj_func = zdt.ZDT2()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT3":
            self.obj_func = zdt.ZDT3()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT4":
            self.obj_func = zdt.ZDT4()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT5":
            self.obj_func = zdt.ZDT5()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT6":
            self.obj_func = zdt.ZDT6()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ1":
            self.obj_func = dtlz.DTLZ1(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ2":
            self.obj_func = dtlz.DTLZ2(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ3":
            self.obj_func = dtlz.DTLZ3(num_of_objectives, num_of_variables)
            self.lower_limits = 0
            self.upper_limits = 1
        elif name == "DTLZ4":
            self.obj_func = dtlz.DTLZ4(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ5":
            self.obj_func = dtlz.DTLZ5(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ6":
            self.obj_func = dtlz.DTLZ6(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ7":
            self.obj_func = dtlz.DTLZ7(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds

    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions."""
        return self.obj_func(decision_variables)

    def constraints(self, decision_variables, objective_variables):
        """Calculate constraint violation."""
        print("Error: Constraints not supported yet.")


class Parameters:
    """This object contains the parameters necessary for evolution."""

    def __init__(
        self,
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = True,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 10,
        Alpha: float = 2,
        plotting: bool = True,
        algorithm_name="RVEA",
        *args
    ):
        """Initialize the parameters class."""
        self.algorithm_name = algorithm_name
        if algorithm_name == "RVEA":
            rveaparams = {
                "population_size": population_size,
                "lattice_resolution": lattice_resolution,
                "algorithm": rvea,
                "interact": interact,
                "a_priori": a_priori_preference,
                "generations": generations_per_iteration,
                "iterations": iterations,
                "Alpha": Alpha,
                "ploton": plotting,
            }
        self.params = rveaparams


class Population:
    """Define the population."""

    def __init__(
        self,
        problem: Problem,
        parameters: Parameters,
        assign_type: str = "RandomAssign",
        *args
    ):
        """Initialize the population.

        Parameters:
        ------------
            problem: An object of the class Problem

            parameters: An object of the class Parameters

            assign_type: Define the method of creation of population.
                If 'assign_type' is 'RandomAssign' the population is generated
                randomly.
                If 'assign_type' is 'LHSDesign', the population is generated
                via Latin Hypercube Sampling.
                If 'assign_type' is 'custom', the population is imported from
                file.
                If assign_type is 'empty', create blank population.

        """
        pop_size = parameters.params["population_size"]
        num_var = problem.num_of_variables
        self.lower_limits = np.asarray(problem.lower_limits)
        self.upper_limits = np.asarray(problem.upper_limits)
        self.hyp = 0
        self.non_dom = 0
        if assign_type == "RandomAssign":
            self.individuals = np.random.random((pop_size, num_var))
            # Scaling
            self.individuals = (
                self.individuals * (self.upper_limits - self.lower_limits)
                + self.lower_limits
            )
        elif assign_type == "LHSDesign":
            self.individuals = lhs(num_var, samples=pop_size)
            # Scaling
            self.individuals = (
                self.individuals * (self.upper_limits - self.lower_limits)
                + self.lower_limits
            )
        elif assign_type == "custom":
            print("Error: Custom assign type not supported yet.")
        elif assign_type == "empty":
            self.individuals = np.asarray([])
            self.objectives = np.asarray([])
            self.fitness = np.asarray([])
            self.constraint_violation = np.asarray([])
        pop_eval = self.evaluate(problem)
        self.objectives = pop_eval["objectives"]
        self.constraint_violation = pop_eval["constraint violation"]
        self.fitness = pop_eval["fitness"]
        self.reference_vectors = None

    def evaluate(self, problem: Problem):
        """Evaluate and return objective values."""
        pop = self.individuals
        objs = None
        cons = None
        for ind in pop:
            if objs is None:
                objs = np.asarray(problem.objectives(ind))
            else:
                objs = np.vstack((objs, problem.objectives(ind)))
        if problem.num_of_constraints:
            for ind, obj in zip(pop, objs):
                if cons is None:
                    cons = problem.constraints(ind, obj)
                else:
                    cons = np.vstack((cons, problem.constraints(ind, obj)))
            fitness = self.eval_fitness(pop, objs, problem)
        else:
            cons = np.zeros((pop.shape[0], 1))
            fitness = objs
        return {"objectives": objs, "constraint violation": cons, "fitness": fitness}

    def eval_fitness(self, pop, objs, problem):
        """Return fitness values. Maybe add maximization support here."""
        fitness = objs
        return fitness

    def add(self, new_pop: np.ndarray, problem: Problem):
        """Evaluate and add individuals to the population."""
        if new_pop.ndim == 1:
            self.append_individual(new_pop, problem)
        elif new_pop.ndim == 2:
            for ind in new_pop:
                self.append_individual(ind, problem)
        else:
            print("Error while adding new individuals. Check dimensions.")

    def keep(self, indices: list):
        """Remove individuals from population which are not in "indices"."""
        new_pop = self.individuals[indices, :]
        new_obj = self.objectives[indices, :]
        new_fitness = self.fitness[indices, :]
        new_CV = self.constraint_violation[indices, :]
        self.individuals = new_pop
        self.objectives = new_obj
        self.fitness = new_fitness
        self.constraint_violation = new_CV

    def append_individual(self, ind: np.ndarray, problem: Problem):
        """Evaluate and add individual to the population."""
        self.individuals = np.vstack((self.individuals, ind))
        obj, CV, fitness = self.evaluate_individual(ind, problem)
        self.objectives = np.vstack((self.objectives, obj))
        self.constraint_violation = np.vstack((self.constraint_violation, CV))
        self.fitness = np.vstack((self.fitness, fitness))

    def evaluate_individual(self, ind: np.ndarray, problem: Problem):
        """
        Evaluate individual.

        Returns objective values, constraint violation, and fitness.
        """
        obj = problem.objectives(ind)
        CV = 0
        fitness = obj
        if problem.num_of_constraints:
            CV = problem.constraints(ind, obj)
            fitness = self.eval_fitness(ind, obj, problem)
        return (obj, CV, fitness)

    def evolve(self, problem: Problem, parameters: Parameters) -> "Population":
        """Evolve and return the population with interruptions."""
        population = self
        self.reference_vectors = ReferenceVectors(
            parameters.params["lattice_resolution"], problem.num_of_objectives
        )
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                isnotebook = True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                isnotebook = False  # Terminal running IPython
            else:
                isnotebook = False  # Other type (?)
        except NameError:
            isnotebook = False
        if parameters.params["ploton"]:
            figure, ax = self.plot_init_()
        if isnotebook:
            progressbar = tqdm_notebook
        else:
            progressbar = tqdm

        iterations = parameters.params["iterations"]
        for i in progressbar(range(iterations), desc="Iteration"):
            self = parameters.params["algorithm"](
                self, problem, parameters.params, self.reference_vectors, progressbar
            )
            if parameters.params["ploton"]:
                population.plot_objectives(figure, ax)
            interrupt_evolution(self.reference_vectors, population, problem, parameters)
        return population

    def mate(self):
        """
        Conduct crossover and mutation over the population.

        Conduct simulated binary crossover and bounded polunomial mutation.
        Return offspring population as an array.
        """
        pop = self.individuals
        pop_size, num_var = pop.shape
        shuffled_ids = list(range(pop_size))
        shuffle(shuffled_ids)
        mating_pop = pop[shuffled_ids]
        if pop_size % 2 == 1:
            # Maybe it should be pop_size-1?
            mating_pop = np.vstack((mating_pop, mating_pop[0]))
            pop_size = pop_size + 1
        # The rest closely follows the matlab code.
        ProC = 1
        ProM = 1 / num_var
        DisC = 30
        DisM = 20
        offspring = np.zeros_like(mating_pop)  # empty_like() more efficient?
        for i in range(0, pop_size, 2):
            beta = np.zeros(num_var)
            miu = np.random.rand(num_var)
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
            beta[np.random.rand(num_var) > ProC] = 1  # It was in matlab code
            avg = (mating_pop[i] + mating_pop[i + 1]) / 2
            diff = (mating_pop[i] - mating_pop[i + 1]) / 2
            offspring[i] = avg + beta * diff
            offspring[i] = avg - beta * diff
        min_val = np.ones_like(offspring) * self.lower_limits
        max_val = np.ones_like(offspring) * self.upper_limits
        k = np.random.random(offspring.shape)
        miu = np.random.random(offspring.shape)
        temp = np.logical_and((k <= ProM), (miu < 0.5))
        offspring_scaled = (offspring - min_val) / (max_val - min_val)
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                (
                    2 * miu[temp]
                    + (1 - 2 * miu[temp]) * (1 - offspring_scaled[temp]) ** (DisM + 1)
                )
                ** (1 / (DisM + 1))
                - 1
            )
        )
        temp = np.logical_and((k <= ProM), (miu >= 0.5))
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                1
                - (
                    2 * (1 - miu[temp])
                    + 2 * (miu[temp] - 0.5) * offspring_scaled[temp] ** (DisM + 1)
                )
                ** (1 / (DisM + 1))
            )
        )
        offspring[offspring > max_val] = max_val[offspring > max_val]
        offspring[offspring < min_val] = min_val[offspring < min_val]
        return offspring

    def plot_init_(self):
        """Initialize plot objects."""
        obj = self.objectives
        num_obj = obj.shape[1]
        if num_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=Axes3D.name)
        else:
            fig, ax = plt.subplots()
        plt.ion()
        plt.show()
        self.plot_objectives(fig, ax)
        return (fig, ax)

    def plot_objectives(self, fig, ax):
        """Plot the objective values of individuals in notebook. This is a hack."""
        obj = self.objectives
        ref = self.reference_vectors.values
        num_samples, num_obj = obj.shape
        ax.clear()
        if num_obj == 2:
            plt.scatter(obj[:, 0], obj[:, 1])
        elif num_obj == 3:
            ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2])
            ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2])
        else:
            objectives = pd.DataFrame(obj)
            objectives["why"] = objectives[0]
            color = plt.cm.rainbow(np.linspace(0, 1, len(objectives.index)))
            ax.clear()
            ax = parallel_coordinates(objectives, "why", ax=ax, color=color)
            ax.get_legend().remove()
        fig.canvas.draw()

    def hypervolume(self, ref_point):
        """Calculate hypervolume. Uses package pygmo."""
        non_dom = self.non_dom
        if not isinstance(ref_point, (Sequence, np.ndarray)):
            num_obj = non_dom.shape[1]
            ref_point = [ref_point] * num_obj
        hyp = hv(non_dom)
        self.hyp = hyp.compute(ref_point)
        return self.hyp

    def non_dominated(self):
        """Return the pareto front of a given population."""
        obj = self.objectives
        num_obj = obj.shape[1]
        if num_obj == 2:
            non_dom_front = nd2(obj)
        else:
            non_dom_front = nds(obj)
        self.non_dom = self.objectives[non_dom_front[0][0]]
        return non_dom_front


class ReferenceVectors:
    """Class object for reference vectors."""

    def __init__(self, lattice_resolution: int, number_of_objectives):
        """Create a simplex lattice."""
        number_of_vectors = comb(
            lattice_resolution + number_of_objectives - 1,
            number_of_objectives - 1,
            exact=True,
        )
        temp1 = range(1, number_of_objectives + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, number_of_objectives - 1)))
        temp2 = np.array([range(number_of_objectives - 1)] * number_of_vectors)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_vectors, number_of_objectives), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, number_of_objectives - 1):
            weight[:, i] = temp[:, i] - temp[:, i - 1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        self.values = weight / lattice_resolution
        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_vectors = number_of_vectors
        self.normalize()
        self.initial_values = self.values[:]
        # self.iteractive_adapt_1() Can use this for a priori preferences!

    def normalize(self):
        """Normalize the reference vectors."""
        self.number_of_vectors = self.values.shape[0]
        norm = np.linalg.norm(self.values, axis=1)
        norm = np.repeat(norm, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives
        )
        self.values = np.divide(self.values, norm)

    def neighbouring_angles(self) -> np.ndarray:
        """Calculate neighbouring angles for normalization."""
        cosvv = np.dot(self.values, self.values.transpose())
        cosvv.sort(axis=1)
        cosvv = np.flip(cosvv, 1)
        acosvv = np.arccos(cosvv[:, 1])
        return acosvv

    def adapt(self, fitness):
        """Adapt reference vectors."""
        max_val = np.amax(fitness, axis=0)
        min_val = np.amin(fitness, axis=0)
        self.values = np.multiply(
            self.initial_values,
            np.tile(np.subtract(max_val, min_val), (self.number_of_vectors, 1)),
        )
        self.normalize()

    def iteractive_adapt_1(self, ref_point, translation_param=0.2):
        """Adapt reference vectors linearly towards a reference point.

        The details can be found in the following paper: Hakanen, Jussi & Chugh, Tinkle
        & Sindhya, Karthik & Jin, Yaochu & Miettinen, Kaisa. (2016). Connections of
        Reference Vectors and Different Types of Preference Information in
        Interactive Multiobjective Evolutionary Algorithms.

        Parameters:
        ------------
            ref_point: list. Signifies the reference point towards which the
                       reference vectors are translated.
            translation_param: double between 0 and 1.
                               Describes the strength of translation.

        """
        self.values = self.initial_values * translation_param + (
            (1 - translation_param) * ref_point
        )
        self.normalize()

    def add_edge_vectors(self):
        """Add edge vectors to the list of reference vectors.

        Used to cover the entire orthant when preference information is provided.
        """
        edge_vectors = np.eye(self.values.shape[1])
        self.values = np.vstack([self.values, edge_vectors])
        self.number_of_vectors = self.values.shape[0]
        self.normalize()

    def plot_ref_V(self):
        """Plot the reference vectors."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)
        plt.ion()
        plt.show()
        ax.scatter(self.values[:, 0], self.values[:, 1], self.values[:, 2])
        ax.scatter(
            self.initial_values[:, 0],
            self.initial_values[:, 1],
            self.initial_values[:, 2],
            alpha=0.5
        )
        ax.view_init(azim=45)
        fig.canvas.draw()
        plt.show(block=True)


def interrupt_evolution(
    reference_vectors: ReferenceVectors,
    population: Population,
    problem: Problem = None,
    parameters: Parameters = None,
):
    """Perform operations while optimization is interrupted.

    Currently supported: Adaptaion of reference vectors with or without preference info.

    Parameters
    ----------
    reference_vectors: ReferenceVectors Object

    population: A Population Object
    problem: Object of the class Problem or derived from class Problem.

    """
    if parameters.algorithm_name == "RVEA":
        if parameters.params["interact"] or parameters.params["a_priori"]:
            # refpoint = np.mean(population.fitness, axis=0)
            ideal = np.amin(population.fitness, axis=0)
            nadir = np.amax(population.fitness, axis=0)
            refpoint = np.zeros_like(ideal)
            print('Ideal vector is ', ideal)
            print('Nadir vector is ', nadir)
            for index in range(len(refpoint)):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    print("Nadir value = ", nadir[index])
                    pref_val = float(
                        input("Please input a value between ideal and nadir: ")
                    )
                    if pref_val > ideal[index] and pref_val < nadir[index]:
                        refpoint[index] = pref_val
                        break
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            reference_vectors.iteractive_adapt_1(refpoint)
            reference_vectors.add_edge_vectors()
        else:
            reference_vectors.adapt(population.fitness)
    elif parameters.algorithm_name == "KRVEA":
        reference_vectors.adapt(population.fitness)
        problem.update(population)


class KrigingProblem(Problem):
    """Create and update kriging models."""

    def __init__(self, dataset, otherarguments):
        """Create a kriging model on the dataset."""
        super.__init__()
        pass

    def objectives(self, decision_variables):
        """Return objective values based on decision variables."""
        pass

    def update(self, population: Population):
        """Update the kriging model based on population.

        Change the return of method objectives.
        """
        pass
