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
    validate_ref_point_with_ideal_and_nadir,
)
# need to add preference_indicator
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

# test with warnings enabled aswell
#np.seterr(all='warn')


class BaseIndicatorEA(BaseEA):
    """The Base class for indicator based EAs.

    This class contains most of the code to set up the parameters and operators.
    It also contains the logic of a indicator EA.

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
        reference_point: np.ndarray = None, # only for PBEA
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
        self.reference_point = reference_point

        if initial_population is not None:
            self.population = initial_population
        elif initial_population is None:
            if population_size is None:
                population_size = 100 
            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size
        

        #print("Using BaseIndicatorEA init")
        
    def start(self):
        return self.requests() 

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


    def _next_gen(self):
        # call _fitness_assigment 
        self._fitness_assignment()
        # iterate until size of new and old population less than old population.
        while (self.population.pop_size < self.population.individuals.shape[0]):
            # choose individual with smallest fitness value, enviromentalSelection
            selected = self._select()
            worst_index = selected

            # update the fitness values
            poplen = self.population.individuals.shape[0]
            for i in range(poplen):
                if self.reference_point is not None: 
                    # over and underflow sometimes, when first run calling this with ibea
                    self.population.fitness[i] += np.exp(-self.indicator(self.population.objectives[i], self.population.objectives[worst_index], self.reference_point, self.delta) / self.kappa)
                else:
                    self.population.fitness[i] += np.exp(-self.indicator(self.population.objectives[i], self.population.objectives[worst_index]) / self.kappa)

            # remove the worst individual 
            self.population.delete(selected)

        # perform binary tournament selection. in these steps 5 and 6 we give offspring to the population and make it bigger. 
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
    def _fitness_assignment(self):
        population = self.population
        pop_size = population.individuals.shape[0]
        pop_width = population.fitness.shape[1]

        for i in range(pop_size):
            population.fitness[i] = [0]*pop_width # 0 all the fitness values. 
            for j in range(pop_size):
                if j != i:
                    if self.reference_point is not None:
                        # sometimes over or underflows, same as the other
                        population.fitness[i] += -np.exp(-self.indicator(population.objectives[i], population.objectives[j], self.reference_point, self.delta) / self.kappa)
                    else:
                        population.fitness[i] += -np.exp(-self.indicator(population.objectives[i], population.objectives[j]) / self.kappa)
                    



    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        """
        # start only with reference point reference as in article
        # check if ibea don't go, also set n iters.TODO: better solution
        # now the iterations make sense.
        #n_iters = 0
        #print("n_it", n_iters)
        if (self.interact is False): 
            #n_iters = self.n_iterations
            #print("n_it", n_iters)
            return

        if preference is None:
            msg = "Giving preferences is mandatory"
            raise eaError(msg)

        if not isinstance(preference, ReferencePointPreference):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)}\n"
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

        if preference is not None:
            self.reference_point = preference.response.values * self.population.problem._max_multiplier
            #self._iteration_counter = 1 # solution on how to keep iterating
            # TODO: bug with calling this 2nd time here hence doubling the n gen per iters every time managing pref
            #self.n_iterations += self.n_iterations # this is second time this is called so thats why they are doubling
            self.n_iterations += self.n_iterations # now only adding the original iterations per dms preference run
            self.total_function_evaluations += self.total_function_evaluations
            print("Reference point", self.reference_point)
            print(self.n_iterations)



    def request_preferences(self) -> ReferencePointPreference:
        # check that if ibea no preferences
        if (self.interact is False): return None

        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        message = ("Provide preference point. TODO add more info")

        def validator(dimensions_data: pd.DataFrame, reference_point: pd.DataFrame):
            validate_ref_point_dimensions(dimensions_data, reference_point)
            validate_ref_point_data_type(reference_point)
            validate_ref_point_with_ideal(dimensions_data, reference_point)
            return
                   
        interaction_priority = "required"
        self._interaction_request_id = np.random.randint(0, 1e9)

        return ReferencePointPreference(
                dimensions_data=dimensions_data,
                message=message,
                interaction_priority=interaction_priority,
                preference_validator=validate_ref_point_with_ideal_and_nadir,
                request_id=self._interaction_request_id,
            
        )


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
