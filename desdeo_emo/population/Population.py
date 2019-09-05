import numpy as np
from typing import Union, List, Dict
from abc import ABC, abstractmethod
from desdeo_problem.Problem import MOProblem
from desdeo_emo.population.CreateIndividuals import create_new_individuals


class BasePopulation(ABC):
    def __init__(self, problem: MOProblem, pop_size: int, pop_params: Dict = None):
        self.pop_size: int = pop_size
        self.problem = problem
        self.individuals: Union[List, np.ndarray] = None
        self.objectives: np.ndarray = None
        self.uncertainity: np.ndarray = None
        self.fitness: np.ndarray = None
        if not problem.n_of_constraints == 0:
            self.constraint_violation = None
        self.ideal = problem.ideal

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
    def __init__(self, problem: MOProblem, pop_size: int, pop_params: Dict = None):
        super().__init__()
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
        self.add(individuals)

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
        objs, cons = self.problem.evaluate(offsprings)
        new_fiteness = objs  # TODO Calculate fitness based on min/maximization criteria
        first_offspring_index = self.individuals.shape[0]
        self.individuals = np.vstack((self.individuals, offsprings))
        self.objectives = np.vstack((self.objectives, objs))
        self.fitness = np.vstack(self.fitness, new_fiteness)
        if self.problem.n_of_constraints != 0:
            self.constraint_violation = np.vstack(
                (self.constraint_violation, cons)
            )
        last_offspring_index = self.individuals.shape[0]
        return list(range[first_offspring_index, last_offspring_index])

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
        if len(self.problem.n_of_constraints) > 0:
            self.constraint_violation = self.constraint_violation[mask]

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
        self.objectives = self.objectives[mask]
        self.fitness = self.fitness[mask]
        if len(self.problem.n_of_constraints) > 0:
            self.constraint_violation = self.constraint_violation[mask]

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
