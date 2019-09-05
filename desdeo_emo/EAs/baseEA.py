import numpy as np
from desdeo_problem.Problem import MOProblem
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from typing import Type, Dict
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors


class BaseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__(self):
        """Initialize EA here. Set up parameters, create EA specific objects."""
        pass

    def set_params(self):
        """Set up the parameters. Save in self.params"""
        pass

    def _next_gen(self):
        """Run one generation of an EA. Change nothing about the parameters."""
        pass

    def _next_iteration(self):
        """Run one iteration (a number of generations) of EA."""
        pass

    def _run_interruption(self):
        """Run interruptions in between iterations. You can update parameters/EA
        specific classes here.
        """
        pass


class BaseDecompositionEA(BaseEA):
    """This class provides the basic structure for decomposition based
    Evolutionary algorithms, such as RVEA or NSGA-III.
    """

    def __init__(
        self,
        problem: MOProblem,
        selection_operator: Type[SelectionBase],
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        selection_parameters: Dict = None,
    ):
        """Initialize a Base Decomposition EA.

        This will call methods to set up the parameters of RVEA, create
        Reference Vectors, and (as of Feb 2019) run the first iteration of RVEA.

        Parameters
        ----------
        population : Population
            This variable is updated as evolution takes place
        EA_parameters : dict
            Takes the EA-specific parameters. This does not include parameters for
                selection or crossover operators.

        Returns
        -------
        Population:
            Returns the Population after evolution.
        """
        lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
        if problem.n_of_objectives < 11:
            lattice_resolution = lattice_res_options[problem.n_of_objectives - 2]
        else:
            lattice_resolution = 3
        self.reference_vectors = ReferenceVectors(
            lattice_resolution, problem.n_of_objectives
        )
        if initial_population is not None:
            #  Population should be compatible.
            self.population = initial_population  # TODO put checks here.
        elif initial_population is None:
            if population_size is None:
                pop_size_options = [50, 105, 120, 126, 132, 112, 156, 90, 275]
                population_size = pop_size_options[problem.num_of_objectives - 2]
            self.population = Population(problem, population_size, population_params)
        self.a_priori = a_priori
        self.interact = interact
        self.n_iterations = n_iterations
        self.n_gen_per_iter = n_gen_per_iter
        self.selection_operator = selection_operator(selection_parameters)
        # print("Using BaseDecompositionEA init")
        self._next_iteration()

    def _next_iteration(self, population: Population):
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.

        Parameters
        ----------
        population : Population
            Contains current population
        """
        self.params["current_iteration_gen_count"] = 1
        while self.continue_iteration():
            self._next_gen(population)
            self.params["current_iteration_gen_count"] += 1
            self.params["current_total_gen_count"] += 1
        self.params["current_iteration_count"] += 1

    def _next_gen(self, population: Population):
        """Run one generation of decomposition based EA.

        This method leaves method.params unchanged. Intended to be used by
        next_iteration.

        Parameters
        ----------
        population: Population
            Population object
        """

        offspring = population.mate(params=self.params)
        population.add(offspring)
        selected = self.select(population)
        population.delete(selected, preserve=True)

    def _run_interruption(self, population: "Population"):
        """Run the interruption phase of RVEA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors, conducts interaction with the user.

        Parameters
        ----------
        population : Population
        """
        if self.params["interact"] or (
            self.params["a_priori"] and self.params["current_iteration_count"] == 1
        ):
            ideal = population.ideal_fitness
            refpoint = np.zeros_like(ideal)
            print("Ideal vector is ", ideal)
            for index in range(len(refpoint)):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    pref_val = float(
                        input("Please input a value worse than the ideal: ")
                    )
                    if pref_val > ideal[index]:
                        refpoint[index] = pref_val
                        break
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            self.params["reference_vectors"].iteractive_adapt_1(refpoint)
            self.params["reference_vectors"].add_edge_vectors()
        else:
            self.params["reference_vectors"].adapt(population.fitness)
        self.params["reference_vectors"].neighbouring_angles()

    def select(self, population) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Parameters
        ----------
        population : Population
            Contains the current population and problem
            information.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population, self.reference_vectors)

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        pass
