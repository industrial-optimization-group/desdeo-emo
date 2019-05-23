from collections import defaultdict
from collections.abc import Sequence
from random import shuffle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyDOE import lhs
from pygmo import fast_non_dominated_sorting as nds
from pygmo import hypervolume as hv
from pygmo import non_dominated_front_2d as nd2
from tqdm import tqdm, tqdm_notebook
from pyrvea.OtherTools.plotlyanimate import animate_init_, animate_next_
from pyrvea.OtherTools.IsNotebook import IsNotebook
from pyrvea.Population.Population import Population
from scipy.stats import bernoulli as bn

if TYPE_CHECKING:
    from pyrvea.Problem.baseProblem import baseProblem
    from pyrvea.EAs.baseEA import BaseEA


class PopulationEvoNN():
    """Define the population."""

    def __init__(
        self,
        problem: "EvoNNProblem",
        assign_type: str = "RandomDesign",
        plotting: bool = False,
        *args
    ):
        """Initialize the population.

        Parameters
        ----------
        problem : baseProblem
            An object of the class Problem
        assign_type : str, optional
            Define the method of creation of population.
            If 'assign_type' is 'RandomDesign' the population is generated
            randomly. If 'assign_type' is 'LHSDesign', the population is
            generated via Latin Hypercube Sampling. If 'assign_type' is
            'custom', the population is imported from file. If assign_type
            is 'empty', create blank population. (the default is "RandomAssign")
        plotting : bool, optional
            (the default is True, which creates the plots)

        """
        """Initialize the population.

        Parameters
        ----------
        problem : baseProblem
            An object of the class Problem
        assign_type : str, optional
            Define the method of creation of population.
            If 'assign_type' is 'RandomDesign' the population is generated
            randomly. If 'assign_type' is 'LHSDesign', the population is
            generated via Latin Hypercube Sampling. If 'assign_type' is
            'custom', the population is imported from file. If assign_type
            is 'empty', create blank population. (the default is "RandomAssign")
        plotting : bool, optional
            (the default is True, which creates the plots)

        """
        num_var = problem.num_of_variables
        self.lower_limits = np.asarray(problem.lower_limits)
        self.upper_limits = np.asarray(problem.upper_limits)
        self.hyp = 0
        self.non_dom = 0
        self.problem = problem
        self.filename = problem.name + "_" + str(problem.num_of_objectives)
        self.plotting = plotting
        # These attributes contain the solutions.
        self.individuals = np.empty((0, self.problem.num_input_nodes, self.problem.num_hidden_nodes), float)
        self.objectives = np.empty((0, self.problem.num_of_objectives), float)
        self.fitness = np.empty((0, self.problem.num_of_objectives), float)
        self.constraint_violation = np.empty(
            (0, self.problem.num_of_constraints), float)
        self.archive = pd.DataFrame(
            columns=["generation", "decision_variables", "objective_values"]
        )
        self.ideal_fitness = np.full((1, self.problem.num_of_objectives), np.inf)
        self.worst_fitness = -1 * self.ideal_fitness

        if not assign_type == "empty":
            self.create_new_individuals(assign_type)

    def create_new_individuals(
        self, design: str = "RandomDesign", pop_size: int = None, decision_variables=None
    ):
        """Create, evaluate and add new individuals to the population. Initiate Plots.

        The individuals can be created randomly, by LHS design, or can be passed by the
        user.

        Parameters
        ----------
        design : str, optional
            Describe the method of creation of new individuals.
            "RandomDesign" creates individuals randomly.
        pop_size : int, optional
            Number of individuals in the population. If none, some default population
            size based on number of objectives is chosen.
        decision_variables : numpy array or list, optional
            Pass decision variables to be added to the population.
        """
        # What is this?
        if decision_variables is not None:
            pass

        if pop_size is None:
            pop_size_options = [50, 105, 120, 126, 132, 112, 156, 90, 275]
            pop_size = pop_size_options[self.problem.num_of_objectives - 2]
        # Create new individuals

        if design == "RandomDesign":
            #individuals = np.random.random((pop_size, self.problem.num_hidden_nodes, self.problem.num_input_nodes+1))
            individuals = np.random.uniform(
                self.problem.w_low, self.problem.w_high, size=(pop_size, self.problem.num_input_nodes, self.problem.num_hidden_nodes)
            )

        else:
            print("Design not yet supported.")

        self.add(individuals)

        if self.plotting:
            self.figure = []
            self.plot_init_()

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

        if new_pop.shape[0] >= 1:
            for i in range(0, new_pop.shape[0]):
                self.append_individual(new_pop[i, :, :])
        else:
            print("Error while adding new individuals. Check dimensions.")
        # print(self.ideal_fitness)
        self.update_ideal_and_nadir()

    def keep(self, indices: list):
        """Remove individuals from population which are NOT in "indices".

        Parameters
        ----------
        indices: list
            Indices of individuals to keep
        """

        new_pop = self.individuals[indices, :, :]
        new_obj = self.objectives[indices, :]
        new_fitness = self.fitness[indices, :]
        new_cv = self.constraint_violation[indices, :]

        self.individuals = new_pop
        self.objectives = new_obj
        self.fitness = new_fitness
        self.constraint_violation = new_cv

    def delete(self, indices: list):
        """Remove individuals from population which ARE in "indices".

        Parameters
        ----------
        indices: list
            Indices of individuals to keep
        """

        new_pop = np.delete(self.individuals, indices, axis=0)
        new_obj = np.delete(self.objectives, indices, axis=0)
        new_fitness = np.delete(self.fitness, indices, axis=0)
        new_cv = np.delete(self.constraint_violation, indices, axis=0)

        self.individuals = new_pop
        self.objectives = new_obj
        self.fitness = new_fitness
        self.constraint_violation = new_cv

    def archive(self, new_pop, new_obj):
        length_of_archive = len(self.archive)
        if length_of_archive == 0:
            gen_count = 0
        else:
            gen_count = self.archive["generation"].iloc[-1] + 1
        new_entries = pd.DataFrame(
            {
                "generation": [gen_count] * len(new_obj),
                "decision_variables": new_pop.tolist(),
                "objective_values": new_obj.tolist(),
            }
        )
        self.archive = self.archive.append(new_entries, ignore_index=True)

    def append_individual(self, ind: np.ndarray):
        """Evaluate and add individual to the population.

        Parameters
        ----------
        ind: np.ndarray
        """
        self.individuals = np.concatenate((self.individuals, [ind]))
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
        CV = np.empty((0, self.problem.num_of_constraints), float)
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
        ea = EA(self, EA_parameters)
        iterations = ea.params["iterations"]
        if self.plotting:
            self.plot_objectives()  # Figure was created in init
        for i in progressbar(range(1, iterations), desc="Iteration"):
            ea._run_interruption(self)
            ea._next_iteration(self)
            if self.plotting:
                self.plot_objectives()

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

    def plot_objectives(self, iteration: int = None):
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
        non_dom = non_dom[np.all(non_dom < ref_point, axis=1), :]
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
            check_ideal_with = self.fitness
        else:
            check_ideal_with = new_objective_vals
        self.ideal_fitness = np.amin(
            np.vstack((self.ideal_fitness, check_ideal_with)), axis=0
        )
        self.worst_fitness = np.amax(
            np.vstack((self.worst_fitness, check_ideal_with)), axis=0
        )
