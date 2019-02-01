from collections import Sequence
from random import shuffle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from pyDOE import lhs
from pygmo import fast_non_dominated_sorting as nds
from pygmo import hypervolume as hv
from pygmo import non_dominated_front_2d as nd2
from pyRVEA.allclasses import interrupt_evolution
from tqdm import tqdm, tqdm_notebook
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors

if TYPE_CHECKING:
    from pyRVEA.Problem.baseProblem import baseProblem
    from pyRVEA.allclasses import Parameters


class Population:
    """Define the population."""

    def __init__(
        self,
        problem: 'baseProblem',
        parameters: 'Parameters',
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

    def evaluate(self, problem: 'baseProblem'):
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

    def add(self, new_pop: np.ndarray, problem: 'baseProblem'):
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

    def append_individual(self, ind: np.ndarray, problem: 'baseProblem'):
        """Evaluate and add individual to the population."""
        self.individuals = np.vstack((self.individuals, ind))
        obj, CV, fitness = self.evaluate_individual(ind, problem)
        self.objectives = np.vstack((self.objectives, obj))
        self.constraint_violation = np.vstack((self.constraint_violation, CV))
        self.fitness = np.vstack((self.fitness, fitness))

    def evaluate_individual(self, ind: np.ndarray, problem: 'baseProblem'):
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

    def evolve(self, problem: 'baseProblem', parameters: 'Parameters') -> "Population":
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
            offspring[i + 1] = avg - beta * diff
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
