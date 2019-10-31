from typing import Dict, Type

import numpy as np

from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_problem.Problem import MOProblem

class eaError(Exception):
    """Raised when an error related to EA occurs
    """


class BaseEA:
    """This class provides the basic structure for Evolutionary algorithms."""

    def __init__(self):
        """Initialize EA here. Set up parameters, create EA specific objects."""

    def set_params(self):
        """Set up the parameters. Save in self.params"""

    def _next_gen(self):
        """Run one generation of an EA. Change nothing about the parameters."""

    def iterate(self):
        """Run one iteration (a number of generations) of EA."""

    def run_interruption(self):
        """Run interruptions in between iterations. You can update parameters/EA
        specific classes here.
        """


class BaseDecompositionEA(BaseEA):
    """The Base class for decomposition based EAs.

    This class contains most of the code to set up the parameters and operators.
    It also contains the logic of a simple decomposition EA.
    
    Parameters
    ----------
    problem : MOProblem
        The problem class object specifying the details of the problem.
    selection_operator : Type[SelectionBase], optional
        The selection operator to be used by the EA, by default None.
    population_size : int, optional
        The desired population size, by default None, which sets up a default value
        of population size depending upon the dimensionaly of the problem.
    population_params : Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population : Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    lattice_resolution : int, optional
        The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    a_priori : bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact : bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    n_iterations : int, optional
        The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter : int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(
        self,
        problem: MOProblem,
        selection_operator: Type[SelectionBase] = None,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
    ):
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
                population_size = pop_size_options[problem.n_of_objectives - 2]
            self.population = Population(problem, population_size, population_params)
        self.a_priori: bool = a_priori
        self.interact: bool = interact
        self.n_iterations: int = n_iterations
        self.n_gen_per_iter: int = n_gen_per_iter
        self.total_gen_count: int = n_gen_per_iter * n_iterations
        self.total_function_evaluations = total_function_evaluations
        self.selection_operator = selection_operator
        # Internal counters
        self._iteration_counter: int = 0
        self._gen_count_in_curr_iteration = 0
        self._current_gen_count = 0
        self._function_evaluation_count = population_size
        # print("Using BaseDecompositionEA init")

    def iterate(self):
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.
        """
        self._gen_count_in_curr_iteration = 0
        while self.continue_iteration():
            self._next_gen()
        self._iteration_counter += 1

    def _next_gen(self):
        """Run one generation of decomposition based EA. Intended to be used by
        next_iteration.
        """
        offspring = self.population.mate()  # (params=self.params)
        self.population.add(offspring)
        selected = self._select()
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]
            

    def run_interruption(self):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        if self.interact or (self.a_priori and self._iteration_counter == 0):
            ideal = self.population.ideal_fitness_val
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
            self.reference_vectors.iteractive_adapt_1(refpoint)
            self.reference_vectors.add_edge_vectors()
        else:
            self.reference_vectors.adapt(self.population.fitness)
        self.reference_vectors.neighbouring_angles()

    def _select(self) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population, self.reference_vectors)

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return (
            self._gen_count_in_curr_iteration < self.n_gen_per_iter
            and self.check_FE_count()
        )

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        return self._iteration_counter < self.n_iterations and self.check_FE_count()

    def check_FE_count(self) -> bool:
        """Checks whether termination criteria via function evaluation count has been
            met or not.
        
        Returns:
            bool: True is function evaluation count limit NOT met.
        """
        if self.total_function_evaluations == 0:
            return True
        elif self._function_evaluation_count <= self.total_function_evaluations:
            return True
        return False
