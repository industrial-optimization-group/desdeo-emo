from typing import Dict, Type, Union, Tuple, Callable
import numpy as np
import pandas as pd

from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase

import matplotlib.pyplot as plt
from desdeo_problem import DataProblem, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs import BaseEA
from numba import njit 
import hvwfg as hv
from desdeo_emo.selection.EnvironmentalSelection import EnvironmentalSelection
from desdeo_emo.selection.tournament_select import tour_select
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    PreferredSolutionPreference,
    NonPreferredSolutionPreference,
    BoundPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)
from desdeo_tools.utilities.quality_indicator import epsilon_indicator, epsilon_indicator_ndims

from desdeo_emo.EAs.BaseEA import eaError


# where to put this?
def binary_tournament_select(population:Population) -> list:
        parents = []
        for i in range(int(population.pop_size)): 
            parents.append(
                np.asarray(
                    tour_select(population.fitness[:, 0], 2), 
                    tour_select(population.fitness[:, 0], 2),
            ))
        return parents



class BaseIndicatorEA(BaseEA):
    """

    """
    def __init__(
        self,
        problem: MOProblem,
        selection_operator: Type[SelectionBase] = None,
        population_size: int = None, # size required
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        indicator: Callable = None, 
        reference_point = None, # only for PBEA
    ):
        super().__init__(
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            selection_operator=selection_operator,
            use_surrogates=use_surrogates,
        )

        self.indicator = indicator

        if initial_population is not None:
            self.population = initial_population
        elif initial_population is None:
            if population_size is None:
                population_size = 100 
            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size
        
        #if reference_point is None:

        #print("Using BaseIndicatorEA init")
        
    def start(self):
        pass

    def return_pop(self):
        return self.population

    def end(self):
        """Conducts non-dominated sorting at the end of the evolution process
        Returns:
            tuple: The first element is a 2-D array of the decision vectors of the non-dominated solutions.
                The second element is a 2-D array of the corresponding objective values.
        """
        non_dom = self.population.non_dominated_objectives()
        return (
            self.population.individuals[non_dom, :],
            self.population.objectives[non_dom, :],
        )


    # täälä operaattorit in the main loop of the algoritmh
    def _next_gen(self):
        # call _fitness_assigment (using indicator). replacement
        self._fitness_assignment()
        # iterate until size of new and old population less than old population.
        while (self.population.pop_size < self.population.individuals.shape[0]):
            # choose individual with smallest fitness value
            selected = self._select()
            worst_index = selected[0]

            # update the fitness values
            poplen = self.population.individuals.shape[0]
            for i in range(poplen):
                 self.population.fitness[i] += np.exp(-self.indicator(self.population.objectives[i], self.population.objectives[worst_index]) / self.kappa)

            # remove the worst individula 
            self.population.delete(selected)

        # check termination
        if (self._function_evaluation_count >= self.total_function_evaluations):
            # just to stop the iteration. 
            self.end()

        # perform binary tournament selection. in these steps 5 and 6 we give offspring to the population and make it bigger. kovakoodataan tämä nytten, mietitään myöhemmin sitten muuten.
        chosen = binary_tournament_select(self.population)

        # variation, call the recombination operators
        offspring = self.population.mate(mating_individuals=chosen)
        self.population.add(offspring)

        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]


    # need to implement the enviromental selection. Only calls it from selection module
    def _select(self) -> list:
        return self.selection_operator.do(self.population)

    #implements fitness computing. 
    # TODO: trouble of calling different indicators with the design since indicators are just functions. Let's cross that bridge when we come to it.
    def _fitness_assignment(self):
        population = self.population
        pop_size = population.individuals.shape[0]
        pop_width = population.fitness.shape[1]

        for i in range(pop_size):
            population.fitness[i] = [0]*pop_width # 0 all the fitness values. 
            for j in range(pop_size):
                if j != i:
                    #print(self.indicator)
                    population.fitness[i] += -np.exp(-self.indicator(population.objectives[i], population.objectives[j]) / self.kappa)
                    



    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        # start only with reference point reference as in article

        #if (self.__name__ == IBEA): return
        # check if ibea don't go
        if (True): return

        if not isinstance(
            preference,
            (
                ReferencePointPreference,
                #PreferredSolutionPreference,
                #NonPreferredSolutionPreference,
                #BoundPreference,
                type(None),
            ),
        ):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)}\n"
                #f"{type(PreferredSolutionPreference)}\n"
                #f"{type(NonPreferredSolutionPreference)}\n"
                #f"{type(BoundPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)
        if preference is not None:
            if preference.request_id != self._interaction_request_id:
                msg = (
                    f"Wrong preference object sent. Expected id = "
                    f"{self._interaction_request_id}.\n"
                    f"Recieved id = {preference.request_id}"
                )
                raise eaError(msg)
        if preference is None and not self._ref_vectors_are_focused:
            self.reference_vectors.adapt(self.population.fitness)
        if isinstance(preference, ReferencePointPreference):
            ideal = self.population.ideal_fitness_val
            refpoint = (
                preference.response.values * self.population.problem._max_multiplier
            )
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint))) # refpoint is normalized here
            refpoint = refpoint / norm
            self.reference_vectors.iteractive_adapt_3(refpoint)
            self.reference_vectors.add_edge_vectors()
#        elif isinstance(preference, PreferredSolutionPreference):
#            self.reference_vectors.interactive_adapt_1(
#                z=self.population.objectives[preference.response],
#                n_solutions=np.shape(self.population.objectives)[0],
#            )
#            self.reference_vectors.add_edge_vectors()
#        elif isinstance(preference, NonPreferredSolutionPreference):
#            self.reference_vectors.interactive_adapt_2(
#                z=self.population.objectives[preference.response],
#                n_solutions=np.shape(self.population.objectives)[0],
#            )
#            self.reference_vectors.add_edge_vectors()
#        elif isinstance(preference, BoundPreference):
#            self.reference_vectors.interactive_adapt_4(preference.response)
#            self.reference_vectors.add_edge_vectors()
        self.reference_vectors.neighbouring_angles()

    def request_preferences(self) -> Union[
        None,
        Tuple[
            PreferredSolutionPreference,
            #NonPreferredSolutionPreference,
            #ReferencePointPreference,
            #BoundPreference,
        ],
    ]:
        # check that if ibea no preferences
        if (True): return None

    def request_plot(self) -> SimplePlotRequest:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        data = pd.DataFrame(
            self.population.objectives, columns=self.population.problem.objective_names
        )
        return SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="Objective Values"
        )

    def requests(self) -> Tuple:
        return (self.request_preferences(), self.request_plot())
