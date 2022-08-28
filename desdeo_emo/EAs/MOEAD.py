from typing import Dict

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.MOEAD_select import MOEAD_select
from desdeo_problem import MOProblem
from desdeo_problem.problem.Problem import EvaluationResults
from desdeo_tools.scalarization import MOEADSF
from desdeo_tools.scalarization.MOEADSF import PBI


class MOEA_D(BaseDecompositionEA):
    """Python implementation of MOEA/D

    .. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
    in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.

    Parameters
    ----------
    problem: MOProblem
        The problem class object specifying the details of the problem.
    scalarization_function: MOEADSF
        The scalarization function to compare the solutions. Some implementations
        can be found in desdeo-tools/scalarization/MOEADSF. By default it uses the
        PBI function.
    n_neighbors: int, optional
        Number of reference vectors considered in the neighborhoods creation. The default
        number is 20.
    population_params: Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population: Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    lattice_resolution: int, optional
        The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    n_parents: int, optional
        Number of individuals considered for the generation of offspring solutions. The default
        option is 2.
    a_priori: bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact: bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    use_surrogates: bool, optional
        A bool variable defining whether surrogate problems are to be used or
        not. By default False
    n_iterations: int, optional
         The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter: int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations: int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(  # parameters of the class
        self,
        problem: MOProblem,
        n_neighbors: int = 20,
        n_parents: int = 2,
        use_repair: bool = True,
        initial_population: Population = None,
        population_size: int = None,
        population_params: Dict = None,
        lattice_resolution: int = None,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        use_surrogates: bool = False,
        total_function_evaluations: int = 0,
        scalarization_function: MOEADSF = PBI(),
        keep_archive: bool = False,
        save_non_dominated: bool = False,
    ):
        super().__init__(  # parameters for decomposition based approach
            problem=problem,
            population_params=population_params,
            population_size=population_size,
            keep_archive=keep_archive,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            save_non_dominated=save_non_dominated,
        )

        self.use_repair = use_repair

        # New crossover and mutation operators needed for more than two parents.
        if n_parents != 2:
            raise ValueError(
                "n_parents must be equal to 2. Other values not yet supported."
            )
        self.n_parents = n_parents

        self.selection_operator = MOEAD_select(
            self.population, SF_type=scalarization_function, n_neighbors=n_neighbors
        )

        # self._ideal_point = self.population.ideal_fitness_val

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population.pop_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            selected_parents = self.selection_operator.choose_parents(
                current_neighborhood=i, n_parents=self.n_parents
            )

            # TODO: Implement better recombination operators for this steps
            # Currently SBX and PolyM are used, they produce two offspring.
            # Apply genetic operators over two random individuals
            offspring = self.population.mate(selected_parents)
            offspring = offspring[0, :]

            # Repair the solution if it is needed
            if self.use_repair:
                offspring = self.population.repair(offspring)

            # Evaluate the offspring by adding it to the population list
            results = self.population.add(offspring, use_surrogates=self.use_surrogates)
            if not self.use_surrogates:
                self._function_evaluation_count += 1

            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood=i)

            self.population.replace(selected, offspring, results)

            # Remove the extra individual added earlier
            self.population.delete([-1])
            if self.save_non_dominated:
                self.non_dominated_archive(offspring, results)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1

    def _select(self, current_neighborhood: int) -> list:
        return self.selection_operator.do(
            self.population,
            current_neighborhood,
        )
