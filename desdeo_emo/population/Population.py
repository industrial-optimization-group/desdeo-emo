from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from desdeo_emo.population.CreateIndividuals import create_new_individuals
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover
from desdeo_problem import MOProblem

from desdeo_tools.utilities import non_dominated


class BasePopulation(ABC):
    def __init__(self, problem: MOProblem, pop_size: int, pop_params: Dict = None):
        self.pop_size: int = pop_size
        self.problem = problem
        self.individuals: Union[List, np.ndarray] = None
        self.objectives: np.ndarray = None
        self.uncertainity: np.ndarray = None
        self.fitness: np.ndarray = None
        if not problem.n_of_constraints == 0:
            self.constraint = None
        self.nadir_objective_vector = problem.nadir
        self.nadir_fitness_val = None
        if problem.ideal is not None:
            self.nadir_fitness_val = problem.nadir * problem._max_multiplier
        self.xover = None
        self.mutation = None
        self.recombination = None

    @property
    def ideal_objective_vector(self) -> np.ndarray:
        return self.problem.ideal

    @property
    def ideal_fitness_val(self) -> np.ndarray:
        return self.problem.ideal_fitness

    @abstractmethod
    def add(self, offsprings: Union[List, np.ndarray]) -> List:
        """Evaluate and add offspring to the population.

        Parameters
        ----------
        offsprings : Union[List, np.ndarray]
            List or array of individuals to be evaluated and added to the population.

        Returns
        -------
        List
            Indices of the evaluated individuals
        """
        pass

    @abstractmethod
    def keep(self, indices: List):
        """Save the population members given by the list of indices for the next
            generation. Delete the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be kept for the next
                generation.
        """
        pass

    @abstractmethod
    def delete(self, indices: List):
        """Delete the population members given by the list of indices for the next
            generation. Keep the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be deleted.
        """
        pass

    @abstractmethod
    def mate(
        self, mating_individuals: List = None, params: Dict = None
    ) -> Union[List, np.ndarray]:
        """Perform crossover and mutation over the population members.

        Parameters
        ----------
        mating_individuals : List, optional
            List of individuals taking part in recombination. By default None, which
                recombinated all individuals in random order.
        params : Dict, optional
            Parameters for the mutation or crossover operator, by default None.

        Returns
        -------
        Union[List, np.ndarray]
            The offspring population
        """
        pass


class Population(BasePopulation):
    def __init__(
        self,
        problem: MOProblem,
        pop_size: int,
        pop_params: Dict = None,
        use_surrogates: bool = False,
    ):
        super().__init__(problem, pop_size)
        self.lower_limits = self.problem.get_variable_lower_bounds()
        self.upper_limits = self.problem.get_variable_upper_bounds()
        if pop_params is None:
            design = "LHSDesign"
        if pop_params is not None:
            if "design" in pop_params.keys():
                design = pop_params["design"]
            else:
                design = "LHSDesign"
        individuals = create_new_individuals(design, problem, pop_size)
        self.add(individuals, use_surrogates)
        self.xover = SBX_xover()
        self.mutation = BP_mutation(self.lower_limits, self.upper_limits)

    def add(self, offsprings: Union[List, np.ndarray], use_surrogates: bool = False):
        """Evaluate and add offspring to the population.

        Parameters
        ----------
        offsprings : Union[List, np.ndarray]
            List or array of individuals to be evaluated and added to the population.

        use_surrogates: bool
            If true, use surrogate models rather than true function evaluations.

        use_surrogates: bool
            If true, use surrogate models rather than true function evaluations.

        Returns
        -------
        Results
            Results of evaluation.
        """
        results = self.problem.evaluate(offsprings, use_surrogates)
        objectives = results.objectives
        fitness = results.fitness
        constraints = results.constraints
        uncertainity = results.uncertainity
        if self.individuals is None:
            self.individuals = offsprings
            self.objectives = objectives
            self.fitness = fitness
            self.constraint = constraints
            self.uncertainity = uncertainity
            first_offspring_index = 0
        else:
            first_offspring_index = self.individuals.shape[0]
            if self.individuals.ndim - offsprings.ndim == 1:
                self.individuals = np.vstack((self.individuals, [offsprings]))
            elif self.individuals.ndim == offsprings.ndim:
                self.individuals = np.vstack((self.individuals, offsprings))
            else:
                pass  # TODO raise error
            self.objectives = np.vstack((self.objectives, objectives))
            self.fitness = np.vstack((self.fitness, fitness))
            if self.problem.n_of_constraints != 0:
                self.constraint = np.vstack((self.constraint, constraints))
            if uncertainity is None:
                self.uncertainity = None
            else:
                self.uncertainity = np.vstack((self.uncertainity, uncertainity))
        last_offspring_index = self.individuals.shape[0]
        self.update_ideal()
        return results

    def keep(self, indices: List):
        """Save the population members given by the list of indices for the next
            generation. Delete the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be kept for the next
                generation.
        """
        mask = np.zeros(self.individuals.shape[0], dtype=bool)
        mask[indices] = True
        self.individuals = self.individuals[mask]
        self.objectives = self.objectives[mask]
        self.fitness = self.fitness[mask]
        if self.uncertainity is not None:
            self.uncertainity = self.uncertainity[mask]
        if self.problem.n_of_constraints > 0:
            self.constraint = self.constraint[mask]

    def delete(self, indices: List):
        """Delete the population members given by the list of indices for the next
            generation. Keep the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be deleted.
        """
        mask = np.ones(self.individuals.shape[0], dtype=bool)
        mask[indices] = False
        self.individuals = self.individuals[mask]
        if len(self.individuals) == 0:
            self.individuals = None
        self.objectives = self.objectives[mask]
        self.fitness = self.fitness[mask]
        if self.uncertainity is not None:
            self.uncertainity = self.uncertainity[mask]
        if self.problem.n_of_constraints > 0:
            self.constraint = self.constraint[mask]

    def mate(self, mating_individuals: List = None) -> Union[List, np.ndarray]:
        """Perform crossover and mutation over the population members.

        Parameters
        ----------
        mating_individuals : List, optional
            List of individuals taking part in recombination. By default None, which
                recombinated all individuals in random order.
        params : Dict, optional
            Parameters for the mutation or crossover operator, by default None.

        Returns
        -------
        Union[List, np.ndarray]
            The offspring population
        """
        if self.recombination is not None:
            offspring = self.recombination.do(self.individuals, mating_individuals)
        else:
            offspring = self.xover.do(self.individuals, mating_individuals)
            offspring = self.mutation.do(offspring)
        return offspring

    def update_ideal(self):
        pass
        """if self.ideal_fitness_val is None:
            self.ideal_fitness_val = np.amin(self.fitness, axis=0)
        else:
            self.ideal_fitness_val = np.amin(
                np.vstack((self.ideal_fitness_val, self.fitness)), axis=0
            )
        self.ideal_objective_vector = (
            self.ideal_fitness_val * self.problem._max_multiplier
        )"""

    def replace(self, indices: List, individual: np.ndarray, evaluation: tuple):
        """Replace the population members given by the list of indices by the given individual and its evaluation.
           Keep the rest of the population unchanged.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be replaced.
        individual: np.ndarray
            Decision variables of the individual that will replace the positions given in the list.
        evaluation: tuple
            Result of the evaluation of the objective function, constraints, etc. obtained using the evaluate method.
        """
        self.individuals[indices, :] = individual
        self.objectives[indices, :] = evaluation.objectives
        self.fitness[indices, :] = evaluation.fitness
        if self.constraint is not None:
            self.constraint[indices, :] = evaluation.constraints
        if self.uncertainity is not None:
            self.uncertainity[indices, :] = evaluation.uncertainity

    def repair(self, individual):
        """Repair the variables of an individual which are not in the boundary defined by the problem
        Parameters
        ----------
        individual :
            Decision variables of the individual.

        Return
        ----------
        The new decision vector with the variables in the boundary defined by the problem
        """
        upper_bounds = self.problem.get_variable_upper_bounds()
        lower_bounds = self.problem.get_variable_lower_bounds()
        upper_bounds_check = np.where(individual > upper_bounds)
        lower_bounds_check = np.where(individual < lower_bounds)
        individual[upper_bounds_check] = upper_bounds[upper_bounds_check]
        individual[lower_bounds_check] = lower_bounds[lower_bounds_check]
        return individual

    def reevaluate_fitness(self):
        self.fitness = self.problem.reevaluate_fitness(self.objectives)

    def non_dominated_fitness(self):
        return non_dominated(self.fitness)

    def non_dominated_objectives(self):
        return non_dominated(self.objectives * self.problem._max_multiplier)
