from typing import Dict, Callable
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA 
from desdeo_tools.utilities.quality_indicator import epsilon_indicator
from desdeo_emo.selection.TournamentSelection import TournamentSelection
from desdeo_problem import MOProblem


class IBEA(BaseIndicatorEA):
    """Python Implementation of IBEA. 

    Most of the relevant code is contained in the super class. This class just assigns
    the EnviromentalSelection operator to BaseIndicatorEA.

    Parameters
    ----------
    problem: MOProblem
        The problem class object specifying the details of the problem.
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
    total_function_evaluations : int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    use_surrogates: bool, optional
    	A bool variable defining whether surrogate problems are to be used or
        not. By default False
    kappa : float, optional
        Fitness scaling value for indicators. By default 0.05.
    indicator : Callable, optional
        Quality indicator to use in indicatorEAs. By default in IBEA this is additive epsilon indicator.

    """
    def __init__(self,
        problem: MOProblem,
        population_size: int, # size required
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        population_params: Dict = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        kappa: float = 0.05, # fitness scaling ratio
        indicator: Callable = epsilon_indicator, # default indicator is epsilon_indicator
        ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            initial_population=initial_population,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )
        self.indicator = indicator
        self.kappa = kappa
        selection_operator = TournamentSelection(self.population, 2)
        self.selection_operator = selection_operator

    
    
    def _fitness_assignment(self):
        """
            Performs the fitness assignment of the individuals.
        """
        for i in range(self.population.individuals.shape[0]):
            self.population.fitness[i] = 0 # 0 all the fitness values. 
            for j in range(self.population.individuals.shape[0]):
                if j != i:
                   self.population.fitness[i] += -np.exp(-self.indicator(self.population.objectives[i], 
                        self.population.objectives[j]) / self.kappa)


    def _environmental_selection(self):
        """
            Selects the worst member of population, then updates the population members fitness values compared to the worst individual.
            Worst individual is removed from the population.
        """
        while (self.population.pop_size < self.population.individuals.shape[0]):
            worst_index = np.argmin(self.population.fitness, axis=0)[0] # gets the index worst member of population
            # updates the fitness values
            for i in range(self.population.individuals.shape[0]):
                if worst_index != i:
                    self.population.fitness[i] += np.exp(-self.indicator(self.population.objectives[i], 
                        self.population.objectives[worst_index]) / self.kappa)
            # remove the worst member from population
            self.population.delete(worst_index)

