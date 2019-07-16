from collections import defaultdict
from collections.abc import Sequence

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pygmo import fast_non_dominated_sorting as nds
from pygmo import hypervolume as hv
from pygmo import non_dominated_front_2d as nd2

from tqdm import tqdm, tqdm_notebook

from pyrvea.Population.create_individuals import create_new_individuals
import plotly
import plotly.graph_objs as go

from pyrvea.OtherTools.plotlyanimate import animate_init_, animate_next_
from pyrvea.OtherTools.IsNotebook import IsNotebook
from pyrvea.Recombination import (
    evodn2_self_adapting,
    evodn2_gaussian,
    evonn_nodeswap_gaussian,
    evonn_nodeswap_self_adapting,
    evonn_mut_gaussian,
    evonn_xover_nodeswap,
    evonn_mut_self_adapting,
    bounded_polynomial_mutation,
    simulated_binary_crossover,
)

if TYPE_CHECKING:
    from pyrvea.Problem.baseProblem import baseProblem
    from pyrvea.EAs.baseEA import BaseEA


class Population:
    """Define the population."""

    def __init__(
        self,
        problem: "baseProblem",
        assign_type: str = "RandomDesign",
        plotting: bool = True,
        pop_size=None,
        recombination_type=None,
        crossover_type="simulated_binary_crossover",
        mutation_type="bounded_polynomial_mutation",
        *args
    ):
        """Initialize the population.

        Attributes
        ----------
        problem : baseProblem
            An object of the class Problem
        assign_type : str, optional
            Define the method of creation of population.
            If 'assign_type' is 'RandomDesign' the population is generated
            randomly. If 'assign_type' is 'LHSDesign', the population is
            generated via Latin Hypercube Sampling. If 'assign_type' is
            'custom', the population is imported from file. If assign_type
            is 'empty', create blank population.
            'EvoNN' and 'EvoDN2' will create neural networks or deep neural networks, respectively,
             for population .
        plotting : bool, optional
            (the default is True, which creates the plots)
        pop_size : int
            Population size
        recombination_type, crossover_type, mutation_type : str
            Recombination functions. If recombination_type is specified, crossover and mutation
            will be handled by the same function. If None, they are done separately.

        """
        self.num_var = problem.num_of_variables
        self.lower_limits = np.asarray(problem.lower_limits)
        self.upper_limits = np.asarray(problem.upper_limits)
        self.hyp = 0
        self.non_dom = 0
        self.pop_size = pop_size
        self.recombination_funcs = {
            "evodn2_self_adapting": evodn2_self_adapting,
            "evodn2_gaussian": evodn2_gaussian,
            "evonn_nodeswap_gaussian": evonn_nodeswap_gaussian,
            "evonn_nodeswap_self_adapting": evonn_nodeswap_self_adapting,
            "evonn_xover_nodeswap": evonn_xover_nodeswap,
            "evonn_mut_gaussian": evonn_mut_gaussian,
            "evonn_mut_self_adapting": evonn_mut_self_adapting,
            "bounded_polynomial_mutation": bounded_polynomial_mutation,
            "simulated_binary_crossover": simulated_binary_crossover
        }
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.recombination_type = recombination_type
        if recombination_type is None:
            self.crossover = self.recombination_funcs[crossover_type]
            self.mutation = self.recombination_funcs[mutation_type]
        else:
            self.recombination = self.recombination_funcs[recombination_type]
        self.problem = problem
        self.filename = (
            problem.name + "_" + str(problem.num_of_objectives)
        )  # Used for plotting
        self.plotting = plotting
        self.individuals = []
        self.objectives = np.empty((0, self.problem.num_of_objectives), float)
        self.fitness = np.empty((0, self.problem.num_of_objectives), float)
        self.constraint_violation = np.empty(
            (0, self.problem.num_of_constraints), float
        )
        self.archive = pd.DataFrame(
            columns=["generation", "decision_variables", "objective_values"]
        )
        self.ideal_fitness = np.full((1, self.problem.num_of_objectives), np.inf)
        self.worst_fitness = -1 * self.ideal_fitness

        if not assign_type == "empty":
            individuals = create_new_individuals(
                assign_type, problem, pop_size=self.pop_size
            )
            self.add(individuals)

        if self.plotting:
            self.figure = []
            self.plot_init_()

    def add(self, new_pop: np.ndarray):
        """Evaluate and add individuals to the population. Update ideal and nadir point.

        Parameters
        ----------
        new_pop: ndarray
            Decision variable values for new population.
        """
        for i in range(len(new_pop)):
            self.append_individual(new_pop[i])

        self.update_ideal_and_nadir()

    def append_individual(self, ind: np.ndarray):
        """Evaluate and add individual to the population.

        Parameters
        ----------
        ind: np.ndarray
        """

        self.individuals.append(ind)
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

    def delete_or_keep(self, indices, delete_or_keep="delete"):
        """Remove from population individuals which are in indices, or
        keep them and remove all others.

        Parameters
        ----------
        indices: list or ndarray
            Indices of individuals to keep
        delete_or_keep: str
            Whether to delete indices from current population, or keep them and delete others
        """

        mask = np.ones(len(self.individuals), dtype=bool)
        mask[indices] = False

        # new_pop = np.delete(self.individuals, indices, axis=0)
        new_pop = np.array(self.individuals)[mask]
        deleted_pop = np.array(self.individuals)[~mask]

        # new_obj = np.delete(self.objectives, indices, axis=0)
        new_obj = self.objectives[mask]
        deleted_obj = self.objectives[~mask]

        # new_fitness = np.delete(self.fitness, indices, axis=0)
        new_fitness = self.fitness[mask]
        deleted_fitness = self.fitness[~mask]

        if len(self.constraint_violation) > 0:
            # new_cv = np.delete(self.constraint_violation, indices, axis=0)
            new_cv = self.constraint_violation[mask]
            deleted_cv = self.constraint_violation[~mask]
        else:
            deleted_cv = self.constraint_violation
            new_cv = self.constraint_violation

        if delete_or_keep == "delete":
            self.individuals = list(new_pop)
            self.objectives = new_obj
            self.fitness = new_fitness
            self.constraint_violation = new_cv

        elif delete_or_keep == "keep":
            self.individuals = list(deleted_pop)
            self.objectives = deleted_obj
            self.fitness = deleted_fitness
            self.constraint_violation = deleted_cv

    def evolve(self, EA: "BaseEA" = None, **kwargs):
        """Evolve the population with interruptions.

        Evolves the population based on the EA sent by the user.

        Parameters
        ----------
        EA: "BaseEA"
            Should be a derivative of BaseEA (Default value = None)
        **kwargs
            Parameters passed to the EA

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
        ea = EA(self, **kwargs)
        iterations = ea.params["iterations"]

        if self.plotting:
            self.plot_objectives()  # Figure was created in init
        for i in progressbar(range(iterations), desc="Iteration"):
            ea._run_interruption(self)
            ea._next_iteration(self)
            if self.plotting:
                self.plot_objectives()

    def mate(self, mating_pop=None, params=None):
        """Conduct crossover and mutation over the population.

        """

        if self.recombination_type is not None:
            offspring = self.recombination.mate(mating_pop, self.individuals, params)
        else:
            offspring = self.crossover.mate(mating_pop, self.individuals, params)
            self.mutation.mutate(
                offspring,
                self.individuals,
                params,
                self.lower_limits,
                self.upper_limits,
            )

        return offspring

    def eval_fitness(self):
        """
        Calculate fitness based on objective values. Fitness = obj if minimized.
        """
        fitness = self.objectives * self.problem.objs
        return fitness

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

    def plot_pareto(self, name, show_all=False):
        """Plot the pareto front.

        Parameters
        ----------
        name : str
            Name to append to the plot filename.
        show_all : bool
            Show all solutions, including those not on the pareto front.

        """
        if name is None:
            name = self.problem.name

        ndf = self.non_dominated()
        pareto = self.objectives[ndf]
        pareto_pop = np.asarray(self.individuals)[ndf].tolist()

        for idx, x in enumerate(pareto_pop):

            for i, y in enumerate(x):
                x[i] = "x" + str(i + 1) + ": " + str(y) + "<br>"
            x.insert(0, "Model " + str(idx))

        trace0 = go.Scatter(
            x=pareto[:, 0],
            y=pareto[:, 1],
            text=pareto_pop,
            hoverinfo="text",
            mode="markers+lines",
        )

        if show_all:
            trace1 = go.Scatter(
                x=self.objectives[:, 0], y=self.objectives[:, 1], mode="markers"
            )

            data = [trace0, trace1]
        else:
            data = [trace0]

        layout = go.Layout(xaxis=dict(title="f1"), yaxis=dict(title="f2"))
        plotly.offline.plot(
            {"data": data, "layout": layout},
            filename=name
            + "pareto"
            + ".html",
            auto_open=True,
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
