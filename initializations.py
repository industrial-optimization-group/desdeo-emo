"""Testing code."""

from itertools import combinations
# from math import sqrt
from random import shuffle
from deap import benchmarks
import numpy as np
from deap.tools import cxSimulatedBinaryBounded, mutPolynomialBounded
from optproblems import dtlz, zdt
from scipy.special import comb
from pyDOE import lhs
from RVEA import rvea
from pygmo import hypervolume as hv
from pygmo import fast_non_dominated_sorting as nds
from pygmo import non_dominated_front_2d as nd2
from matplotlib import pyplot as plt
from collections import defaultdict

class Problem():
    """Defines the problem."""

    def __init__(
            self, name, num_of_variables, upper_limits, lower_limits,
            num_of_objectives, num_of_constraints):
        """Pydocstring is ruthless."""
        self.name = name
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.num_of_constraints = num_of_constraints
        if name == 'ZDT1':
            self.obj_func = zdt.ZDT1()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'ZDT2':
            self.obj_func = zdt.ZDT2()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'ZDT3':
            self.obj_func = zdt.ZDT3()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'ZDT4':
            self.obj_func = zdt.ZDT4()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'ZDT5':
            self.obj_func = zdt.ZDT5()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'ZDT6':
            self.obj_func = zdt.ZDT6()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ1':
            self.obj_func = dtlz.DTLZ1(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ2':
            self.obj_func = dtlz.DTLZ2(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ3':
            self.obj_func = benchmarks.dtlz3
            self.lower_limits = 0
            self.upper_limits = 1
        if name == 'DTLZ4':
            self.obj_func = dtlz.DTLZ4(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ5':
            self.obj_func = dtlz.DTLZ5(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ6':
            self.obj_func = dtlz.DTLZ6(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        if name == 'DTLZ7':
            self.obj_func = dtlz.DTLZ7(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds

    def objectives(self, decision_variables) ->list:
        """Use this method to calculate objective functions."""
        # if self.name == 'ZDT1':
        #    obj1 = decision_variables[0]
        #    g = 1 + (9/29)*sum(decision_variables[1:])
        #    obj2 = g*(1 - sqrt(obj1/g))
        #    return([obj1, obj2])
        if self.name == 'DTLZ3':
            return(self.obj_func(decision_variables, self.num_of_objectives))
        return(self.obj_func(decision_variables))

    def constraints(self, decision_variables, objective_variables):
        """Calculate constraint violation."""
        print('Error: Constraints not supported yet.')


class Parameters():
    """This object contains the parameters necessary for evolution."""

    def __init__(
            self,
            population_size,
            lattice_resolution=None,
            algorithm_name='RVEA',
            *args):
        """Initialize the parameters class."""
        self.algorithm_name = algorithm_name
        if algorithm_name == 'RVEA':
            rveaparams = {'population_size': population_size,
                          'lattice_resolution': lattice_resolution,
                          'algorithm': rvea,
                          'generations': 1000,
                          'Alpha': 2,
                          'refV_adapt_frequency': 0.1}
        self.params = rveaparams


class Population():
    """Define the population."""

    def __init__(self,
                 problem: Problem,
                 parameters: Parameters,
                 assign_type: str='RandomAssign',
                 *args):
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
        pop_size = parameters.params['population_size']
        num_var = problem.num_of_variables
        self.lower_limits = np.asarray(problem.lower_limits)
        self.upper_limits = np.asarray(problem.upper_limits)
        self.hyp = 0
        self.non_dom = 0
        if assign_type == 'RandomAssign':
            self.individuals = np.random.random((pop_size, num_var))
            # Scaling
            self.individuals = (self.individuals *
                                (self.upper_limits - self.lower_limits) +
                                self.lower_limits)
        elif assign_type == 'LHSDesign':
            self.individuals = lhs(num_var, samples=pop_size)
            # Scaling
            self.individuals = (self.individuals *
                                (self.upper_limits - self.lower_limits) +
                                self.lower_limits)
        elif assign_type == 'custom':
            print('Error: Custom assign type not supported yet.')
        elif assign_type == 'empty':
            self.individuals = np.asarray([])
            self.objectives = np.asarray([])
            self.fitness = np.asarray([])
            self.constraint_violation = np.asarray([])
        pop_eval = self.evaluate(problem)
        self.objectives = pop_eval['objectives']
        self.constraint_violation = pop_eval['constraint violation']
        self.fitness = pop_eval['fitness']

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
            for ind, obj in zip(pop, obj):
                if cons is None:
                    cons = problem.constraints(ind, obj)
                else:
                    cons = np.vstack((cons, problem.constraints(ind, obj)))
            fitness = self.eval_fitness(pop, objs, problem)
        else:
            cons = np.zeros((pop.shape[0], 1))
            fitness = objs
        return({"objectives": objs,
                "constraint violation": cons,
                "fitness": fitness})

    def eval_fitness(self, pop, objs, problem):
        """Return fitness values. Maybe add maximization support here."""
        fitness = objs
        return(fitness)

    def add(self, new_pop: np.ndarray, problem: Problem):
        """Evaluate and add individuals to the population."""
        if new_pop.ndim == 1:
            self.append_individual(new_pop, problem)
        elif new_pop.ndim == 2:
            for ind in new_pop:
                self.append_individual(ind, problem)
        else:
            print('Error while adding new individuals. Check dimensions.')

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
        if problem.constraints:
            CV = problem.constraints(ind, obj)
            fitness = self.eval_fitness(ind, obj, problem)
        return(obj, CV, fitness)

    def evolve(self, problem: Problem, parameters: Parameters)-> 'Population':
        """Evolve and return the population."""
        param = parameters.params
        evolved_population = param['algorithm'](self, problem, param)
        return(evolved_population)

    def ievolve(self, problem: Problem, parameters: Parameters)-> 'Population':
        """Evolve and return the population with interruptions."""
        param = parameters.params['RVEA']
        evolved_population = param.algorithm(self, problem, param)
        return(evolved_population)
    
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
            mating_pop = np.vstack(mating_pop, mating_pop[0])
            pop_size = pop_size + 1
        # The rest closely follows the matlab code.
        ProC = 1
        ProM = 1/num_var
        DisC = 30
        DisM = 20
        offspring = np.zeros_like(mating_pop)  # empty_like() more efficient?
        for i in range(0, pop_size, 2):
            beta = np.zeros(num_var)
            miu = np.random.rand(num_var)
            beta[miu <= 0.5] = (2*miu[miu <= 0.5]) ** (1/(DisC + 1))
            beta[miu > 0.5] = (2-2*miu[miu > 0.5]) ** (-1/(DisC + 1))
            beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_var))
            beta[np.random.rand(10) > ProC] = 1  # Why? It was in matlab code
            avg = (mating_pop[i] + mating_pop[i+1]) / 2
            diff = (mating_pop[i] - mating_pop[i+1]) / 2
            offspring[i] = avg + beta * diff
            offspring[i] = avg - beta * diff
        min_val = np.ones_like(offspring) * self.lower_limits
        max_val = np.ones_like(offspring) * self.upper_limits
        k = np.random.random(offspring.shape)
        miu = np.random.random(offspring.shape)
        temp = np.logical_and((k <= ProM), (miu < 0.5))
        offspring_scaled = (offspring - min_val) / (max_val - min_val)
        offspring[temp] = (offspring[temp] + (max_val[temp]-min_val[temp]) *
                           ((2*miu[temp] + (1 - 2*miu[temp]) *
                            (1 - offspring_scaled[temp]) **
                            (DisM + 1)) ** (1 / (DisM + 1)) - 1))
        temp = np.logical_and((k <= ProM), (miu >= 0.5))
        offspring[temp] = (offspring[temp] + (max_val[temp]-min_val[temp]) *
                           (1 - (2 * (1 - miu[temp]) + 2 * (miu[temp] - 0.5) *
                                 offspring_scaled ** (DisM + 1)) **
                            (1 / (DisM + 1))))
        offspring[offspring > max_val] = max_val[offspring > max_val]
        offspring[offspring < min_val] = min_val[offspring < min_val]
        return(offspring)

    def plot_objectives(self):
        """Plot the objective values of non_dominated individuals."""
        print('Plotting not supported yet.')
        obj = self.objectives
        num_obj = obj.shape[1]  # Check
        if num_obj == 2:
            plt.scatter(obj[:, 0], obj[:, 1])
        elif num_obj == 3:
            plt.scatter(obj[:, 0], obj[:, 1], obj[:, 2])
        else:
            print('Plotting more than 3 objectives not supported yet.')

    def hypervolume(self, ref_point):
        """Calculate hypervolume. Uses package pygmo."""
        non_dom = self.non_dom
        if len(ref_point) == 1:
            num_obj = non_dom.shape[1]
            ref_point = [ref_point] * num_obj
        hyp = hv(non_dom)
        self.hyp = hyp.compute(ref_point)
        return(self.hyp)

    def non_dominated(self):
        """Return the pareto front of a given population."""
        obj = self.objectives
        num_obj = obj.shape[1]
        if num_obj == 2:
            non_dom_front = nd2(obj)
        else:
            non_dom_front = nds(obj)
        self.non_dom = non_dom_front
        return(self.non_dom)


