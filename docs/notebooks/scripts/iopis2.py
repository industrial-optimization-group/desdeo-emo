import numpy as np
import pandas as pd


from desdeo_tools.maps import classificationPIS
from desdeo_tools.scalarization import AUG_GUESS_GLIDE

from optproblems import dtlz

from desdeo_problem.Problem import MOProblem
from desdeo_problem.Objective import VectorObjective
from desdeo_problem.Variable import variable_builder

from desdeo_emo.EAs import RVEA, NSGAIII

class classificationPISProblem(MOProblem):

    def __init__(
        self,
        objectives,
        variables,
        nadir: np.ndarray,
        ideal: np.ndarray,
        PIS,
        constraints= None,
    ):
        super().__init__(
            objectives=objectives,
            variables=variables,
            constraints=constraints,
            nadir=nadir,
            ideal=ideal,
        )
        print(np.atleast_2d(self.ideal * self._max_multiplier))
        print(np.atleast_2d(self.nadir * self._max_multiplier))
        self.ideal_fitness = PIS(np.atleast_2d(self.ideal * self._max_multiplier))
        self.nadir_fitness = PIS(np.atleast_2d(self.nadir * self._max_multiplier))
        self.PIS = PIS

        self.num_dim_fitness = len(PIS.scalarizers) + 1

    def evaluate_fitness(self, objective_vectors: np.ndarray) -> np.ndarray:
        return self.PIS(np.atleast_2d(objective_vectors * self._max_multiplier))

    def reevaluate_fitness(self, objective_vectors: np.ndarray) -> np.ndarray:
        fitness = self.PIS(objective_vectors * self._max_multiplier)
        self.ideal_fitness = self.PIS(np.atleast_2d(self.ideal * self._max_multiplier))
        self.update_ideal(objective_vectors, fitness)
        return fitness

    def update_preference(self, preference):
        self.PIS.update_preference(preference)

    def update_ideal(self, objective_vectors: np.ndarray, fitness: np.ndarray):
        self.ideal_fitness = np.amin(np.vstack((self.ideal_fitness, fitness)), axis=0)

        self.ideal = (
            np.amin(
                np.vstack((self.ideal, objective_vectors)) * self._max_multiplier,
                axis=0,
            )
            * self._max_multiplier
        )

    
n_obj = 3
n_var = 10
benchmark = dtlz.DTLZ1(num_objectives=n_obj, num_variables=n_var)

variables = variable_builder(
    names=[f"x{i+1}" for i in range(n_var)],
    initial_values=[0]*n_var,
    lower_bounds=[0]*n_var,
    upper_bounds=[1]*n_var,
)

objectives = [VectorObjective(
    name=[f"f{i+1}" for i in range(n_obj)],
    evaluator=lambda x: np.asarray(list(map(benchmark, x)))
)]


utopian = np.asarray([0,0,0]) - 1e-6
nadir = np.asarray([1,1,1])

PIS = classificationPIS(
    scalarizers=[AUG_GUESS_GLIDE],
    utopian=utopian,
    nadir=nadir
)

first_preference = {
    "classifications":["=",">=","<="],
    "current solution": np.asarray([0.5, 0.5, 0.5]),
    "levels": np.asarray([0.5, 0.8, 0.2]),
}
PIS.update_preference(first_preference)

problem = classificationPISProblem(
    objectives=objectives,
    variables=variables,
    nadir=nadir,
    ideal=utopian,
    PIS=PIS
)