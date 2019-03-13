from collections import Sequence
from random import shuffle
from typing import TYPE_CHECKING

import numpy as np
from pyDOE import lhs
from pygmo import fast_non_dominated_sorting as nds
from pygmo import hypervolume as hv
from pygmo import non_dominated_front_2d as nd2
from tqdm import tqdm, tqdm_notebook

from pyRVEA.OtherTools.plotlyanimate import animate_init_, animate_next_
from pyRVEA.OtherTools.IsNotebook import IsNotebook

if TYPE_CHECKING:
    from pyRVEA.Problem.baseProblem import baseProblem
    from pyRVEA.EAs.baseEA import BaseEA


class Population:
    """Define the population."""

    def __init__(
        self,
        problem: "baseProblem",
        assign_type: str = "RandomAssign",
        plotting: bool = True,
        *args
    ):
        """Initialize the population.

        Parameters
        ----------
        problem : baseProblem
            An object of the class Problem
        assign_type : str, optional
            Define the method of creation of population.
            If 'assign_type' is 'RandomAssign' the population is generated
            randomly. If 'assign_type' is 'LHSDesign', the population is
            generated via Latin Hypercube Sampling. If 'assign_type' is
            'custom', the population is imported from file. If assign_type
            is 'empty', create blank population. (the default is "RandomAssign")
        plotting : bool, optional
            (the default is True, which creates the plots)

        """
        pop_size_options = [50, 105, 120, 126, 132, 112, 156, 90, 275]
        pop_size = pop_size_options[problem.num_of_objectives - 2]
        num_var = problem.num_of_variables
        self.lower_limits = np.asarray(problem.lower_limits)
        self.upper_limits = np.asarray(problem.upper_limits)
        self.hyp = 0
        self.non_dom = 0
        self.problem = problem
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
        pop_eval = self.evaluate()
        self.objectives = pop_eval["objectives"]
        self.constraint_violation = pop_eval["constraint violation"]
        self.fitness = pop_eval["fitness"]
        self.ideal_fitness = np.amin(self.fitness, axis=0)
        self.nadir_fitness = np.amax(self.fitness, axis=0)
        self.filename = problem.name + "_" + str(problem.num_of_objectives)
        self.plotting = plotting
        if self.plotting:
            self.figure = []
            self.plot_init_()

    def evaluate(self):
        """Evaluate and return objective values."""
        pop = self.individuals
        objs = None
        cons = None
        for ind in pop:
            if objs is None:
                objs = np.asarray(self.problem.objectives(ind))
            else:
                objs = np.vstack((objs, self.problem.objectives(ind)))
        if self.problem.num_of_constraints:
            for ind, obj in zip(pop, objs):
                if cons is None:
                    cons = self.problem.constraints(ind, obj)
                else:
                    cons = np.vstack((cons, self.problem.constraints(ind, obj)))
            fitness = self.eval_fitness(pop, objs)
        else:
            cons = np.zeros((pop.shape[0], 1))
            fitness = objs
        return {"objectives": objs, "constraint violation": cons, "fitness": fitness}

    def eval_fitness(self):
        """
        Calculate fitness based on objective values. Fitness = obj if minimized.
        """
        fitness = self.objectives * self.problem.objs
        return fitness

    def add(self, new_pop: np.ndarray):
        """Evaluate and add individuals to the population. Update ideal and nadir point.

        Parameters
        ----------
        new_pop: np.ndarray
            Decision variable values for new population.
        """
        if new_pop.ndim == 1:
            self.append_individual(new_pop)
        elif new_pop.ndim == 2:
            for ind in new_pop:
                self.append_individual(ind)
        else:
            print("Error while adding new individuals. Check dimensions.")
        print(self.ideal_fitness)
        self.update_ideal_and_nadir()

    def keep(self, indices: list):
        """Remove individuals from population which are not in "indices".

        Parameters
        ----------
        indices: list
            Indices of individuals to keep
        """
        new_pop = self.individuals[indices, :]
        new_obj = self.objectives[indices, :]
        new_fitness = self.fitness[indices, :]
        new_CV = self.constraint_violation[indices, :]
        self.individuals = new_pop
        self.objectives = new_obj
        self.fitness = new_fitness
        self.constraint_violation = new_CV

    def append_individual(self, ind: np.ndarray):
        """Evaluate and add individual to the population.

        Parameters
        ----------
        ind: np.ndarray
        """
        self.individuals = np.vstack((self.individuals, ind))
        obj, CV, fitness = self.evaluate_individual(ind)
        self.objectives = np.vstack((self.objectives, obj))
        self.constraint_violation = np.vstack((self.constraint_violation, CV))
        self.fitness = np.vstack((self.fitness, fitness))

    def evaluate_individual(self, ind: np.ndarray):
        """Evaluate individual.

        Returns objective values, constraint violation, and fitness.

        Parameters
        ----------
        ind: np.ndarray
        """
        obj = self.problem.objectives(ind)
        CV = 0
        fitness = obj
        if self.problem.num_of_constraints:
            CV = self.problem.constraints(ind, obj)
            fitness = self.eval_fitness(ind, obj, self.problem)
        return (obj, CV, fitness)

    def evolve(self, EA: "BaseEA" = None, EA_parameters: dict = None) -> "Population":
        """Evolve the population with interruptions.

        Evolves the population based on the EA sent by the user.

        Parameters
        ----------
        EA: "BaseEA"
            Should be a derivative of BaseEA (Default value = None)
        EA_parameters: dict
            Contains the parameters needed by EA (Default value = None)

        """
        ##################################
        # To determine whether running in console or in notebook. Used for TQDM.
        # TQDM will be removed in future generations as number of iterations can vary
        if IsNotebook():
            progressbar = tqdm_notebook
        else:
            progressbar = tqdm
        ####################################
        # A basic evolution cycle. Will be updated to optimize() in future versions.
        ea = EA(self)
        iterations = ea.params["iterations"]
        if self.plotting:
            self.plot_objectives(1)  # Figure was created in init
        for i in progressbar(range(1, iterations), desc="Iteration"):
            ea._run_interruption(self)
            ea._next_iteration(self)
            if self.plotting:
                self.plot_objectives(i + 1)

    def mate(self):
        """Conduct crossover and mutation over the population.

        Conduct simulated binary crossover and bounded polunomial mutation.
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
        """Initialize animation objects. Return figure"""
        obj = self.objectives
        self.figure = animate_init_(obj, self.filename + ".html")
        return self.figure

    def plot_objectives(self, iteration: int):
        """Plot the objective values of individuals in notebook. This is a hack.

        Parameters
        ----------
        iteration: int
            Iteration count.
        """
        obj = self.objectives
        self.figure = animate_next_(
            obj, self.figure, self.filename + ".html", iteration
        )

    def hypervolume(self, ref_point):
        """Calculate hypervolume. Uses package pygmo. Add checks to prevent errors.

        Parameters
        ----------
        ref_point
            

        Returns
        -------

        """
        non_dom = self.non_dom
        if not isinstance(ref_point, (Sequence, np.ndarray)):
            num_obj = non_dom.shape[1]
            ref_point = [ref_point] * num_obj
        hyp = hv(non_dom)
        self.hyp = hyp.compute(ref_point)
        return self.hyp

    def non_dominated(self):
        """Fix this. check if nd2 and nds mean the same thing"""
        obj = self.objectives
        num_obj = obj.shape[1]
        if num_obj == 2:
            non_dom_front = nd2(obj)
        else:
            non_dom_front = nds(obj)
        if isinstance(non_dom_front, tuple):
            self.non_dom = self.objectives[non_dom_front[0][0]]
        elif isinstance(non_dom_front, np.ndarray):
            self.non_dom = self.objectives[non_dom_front]
        else:
            print("Non Dom error Line 285 in population.py")
        return non_dom_front

    def update_ideal_and_nadir(self, new_objective_vals: list = None):
        """Updates self.ideal and self.nadir in the fitness space.

        Uses the entire population if new_objective_vals is none.

        Parameters
        ----------
        new_objective_vals : list, optional
            Objective values for a newly added individual (the default is None, which
            calculated the ideal and nadir for the entire population.)

        """
        if new_objective_vals is None:
            check_ideal_with = np.amin(self.fitness, axis=0)
        else:
            check_ideal_with = new_objective_vals
        self.ideal_fitness = np.amin((self.ideal_fitness, check_ideal_with), axis=0)
        self.nadir_fitness = np.amax(self.fitness)