"""Testing code."""

import numpy as np
from deap import tools
from scipy.special import comb
from itertools import combinations
from math import sqrt
from random import random


class Problem():
    """Defines the problem."""

    def __init__(
            self, name, num_of_variables, upper_limits, lower_limits,
            num_of_objectives, num_of_constraints):
        """Pydocstring is ruthless."""
        self.name = name
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.num_of_constraints = num_of_constraints

    def objectives(self, decision_variables) ->list:
        """Use this method to calculate objective functions."""
        if self.name == 'ZDT1':
            obj1 = decision_variables[0]
            g = 1 + (9/29)*sum(decision_variables[1:])
            obj2 = g*(1 - sqrt(obj1/g))
            return([obj1, obj2])
        return([0])

    def Constraints():
        """Calculate constraint violation."""
        pass


class Parameters():
    """This object contains the parameters necessary for RVEA."""

    def __init__(
            self, population_size, lattice_resolution, generations=100,
            Alpha=2, mut_type='PolyMut', xover_type='SBX'):
        """Initialize the population."""
        self.population_size = population_size
        self.lattice_resolution = lattice_resolution
        self.generations = generations
        self.Alpha = Alpha
        self.refV_adapt_frequency = 0.1
        self.mutation = self.MutationParametersInit(mut_type)  # Place Holder
        self.crossover = self.CrossoverParameters(xover_type)  # Place Holder

    class MutationParameters():
        """This object contains the parameters necessary for mutation.

        Currently supported: Bounded Polynomial mutation as implimented
        in NSGA-II by Deb.

        Parameters
        ----------
        eta: Crowding degree of mutation for polynomial mutation.
        High values results in smaller mutations.

        indpb: Independent probability of mutation.

        """

        def __init__(self, mutation_type, eta=20, indpb=0.3):
            """Define mutation type and parameters."""
            if mutation_type == "PolyMut":
                self.mutType = 'PolyMut'
                self.crowding_degree_of_mutation = eta
                self.independent_probability_of_mutation = indpb
            elif mutation_type == "SelfAdapt":
                print('Error: Self Adaptive Mutation not defined yet.')
            else:
                print('Error: Mutation type not defined.')

    class CrossoverParameters():
        """This object contains the parameters necessary for crossover.

        Currently supported: Simulated Binary Crossover

        """

        def __init__(self, xover_type):
            """Define crossover type and parameters."""
            if xover_type == 'SBX':
                self.xover_type = 'SBX'
                self.crowding_degree_of_crossover = 30
            else:
                print('Error: Crossover type not defined')


class Individual():
    """Defines an individual."""

    def __init__(self, problem, assign_type='RandomAssign', variable_values=0):
        """Initialize an individual with zeros.

        parameter: problem is a Problem object.
        parameter: assignType defines how the individual is created.
        parameter: varuableValues contains the values of the variables.
        """
        self.variables = np.zeros(problem.num_of_variables)
        self.objectives = [0]*problem.num_of_objectives
        self.constraint_violation = [0]

        if assign_type == 'RandomAssign':
            self.random_assign(problem)
        elif assign_type == 'LHSDesign':
            pass
        elif assign_type == 'CustomAssign':
            self.custom_assign(problem, variable_values)
        else:
            print('Error: assignType not defined')

    def random_assign(self, problem):
        """Assign random values to individual."""
        for i in range(len(self.variables)):
            self.variables[i] = random()

    def LHS_assign(self, population, problem):
        """LHS design of experiment for individuals."""
        pass

    def custom_assign(self, problem, variable_values):
        """Use to create custom individuals, such as after mating."""
        self.variables = variable_values

    def evaluate(self, problem):
        """Evaluate the individual.

        Updates self.objectives and self.constraint_violation variables.
        """
        self.objectives = problem.objectives(self.variables)
        if problem.num_of_constraints:
            self.constraint_violation = problem.Constraints(self.variables)
        return(self.objectives + self.constraint_violation)

    def mate(self, other, problem, parameters):
        """Perform Crossover and mutation on parents and return 2 children."""
        crossover_parameters = parameters.crossover_parameters
        mutation_parameters = parameters.mutation_parameters
        # Crossover
        parent1 = self.variables
        parent2 = other.variables
        child1, child2 = np.asarray(
            tools.cxSimulatedBinaryBounded(
                parent1, parent2,
                crossover_parameters.crowding_degree_of_crossover,
                problem.lower_limits, problem.upper_limits))
        # Mutation
        if mutation_parameters.mutType == 'PolyMut':
            child1 = np.asarray(tools.mutPolynomialBounded(
                child1, mutation_parameters.degreeOfMutation,
                problem.lower_limits, problem.upper_limits,
                mutation_parameters.independent_probability_of_mutation))
            child1 = Individual(
                Problem, assign_type='CustomAssign', variable_values=child1)
            child2 = np.asarray(tools.mutPolynomialBounded(
                child2, mutation_parameters.degreeOfMutation,
                problem.lower_limits, problem.upper_limits,
                mutation_parameters.independent_probability_of_mutation))
            child2 = Individual(
                Problem, assign_type='CustomAssign', variable_values=child2)
            return child1, child2


class ReferenceVectors():
    """Class object for reference vectors."""

    def __init__(self, lattice_resolution: int, number_of_objectives):
        """Create a simplex lattice."""
        number_of_vectors = comb(
            lattice_resolution + number_of_objectives - 1,
            number_of_objectives - 1, exact=True)
        temp1 = range(1, number_of_objectives + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, number_of_objectives-1)))
        temp2 = np.array([range(number_of_objectives-1)]*number_of_vectors)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_vectors, number_of_objectives), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, number_of_objectives-1):
            weight[:, i] = temp[:, i] - temp[:, i-1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        self.values = weight/lattice_resolution
        self.initial_values = self.values
        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_vectors = number_of_vectors
        self.normalize()

    def normalize(self):
        """Normalize the reference vectors."""
        norm = np.linalg.norm(self.values, axis=1)
        norm = np.repeat(norm, self.number_of_objectives).reshape(
            self.number_of_vectors, self.number_of_objectives)
        self.values = np.divide(self.values, norm)

    def neighbouring_angles(self) -> np.ndarray:
        """Calculate neighbouring angles for normalization."""
        cosvv = np.dot(self.values, self.values.transpose())
        cosvv.sort(axis=1)
        cosvv = np.flip(cosvv, 1)
        acosvv = np.arccos(cosvv[:, 1])
        return(acosvv)

    def adapt(self, max_val, min_val):
        """Adapt reference vectors."""
        shape_of_thingy = self.initial_values.shape[0]
        self.values = np.multiply(self.initial_values,
                                  np.tile(np.substract(max_val, min_val),
                                          (shape_of_thingy, 1)))
        self.normalize()
        pass
