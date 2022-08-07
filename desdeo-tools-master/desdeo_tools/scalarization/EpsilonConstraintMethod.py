import numpy as np

from desdeo_tools.scalarization import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer
from typing import Optional, Callable, Union


class ECMError(Exception):
    """Raised when an error related to the Epsilon Constraint Method is encountered.
    """


class EpsilonConstraintMethod:
    """A class to represent a class for scalarizing MOO problems using the epsilon
        constraint method.

    Attributes:
        objectives (Callable): Objective functions.
        to_be_minimized (int): Integer representing which objective function
            should be minimized.
        epsilons (np.ndarray): Upper bounds chosen by the decison maker.
                               Epsilon constraint functions are defined in a following form:
                                    f_i(x) <= eps_i
                               If the constraint function is of form
                                    f_i(x) >= eps_i
                               Remember to multiply the epsilon value with -1!
        constraints (Optional[Callable]): Function that returns definitions of other constraints, if existing.
    """

    def __init__(
            self, objectives: Callable, to_be_minimized: int, epsilons: np.ndarray,
            constraints: Optional[Callable]
    ):
        self.objectives = objectives
        self._to_be_minimized = to_be_minimized
        self.epsilons = epsilons
        self.constraints = constraints

    def evaluate_constraints(self, xs) -> np.ndarray:
        """
        Returns values of constraints with given decison variables.

        Args:
            xs (np.ndarray): Decision variables.

        Returns:
            Values of constraint functions (both "original" constraints as well as epsilon constraints)
            in a vector.
        """
        xs = np.atleast_2d(xs)

        # evaluate epsilon constraint function "left-side" values with given decision variables
        epsilon_left_side = np.array(
            [val for nrow, row in enumerate(self.objectives(xs))
             for ival, val in enumerate(row) if ival != self._to_be_minimized
             ])

        if len(epsilon_left_side) != len(self.epsilons):
            msg = ("The lenght of the epsilons array ({}) must match the total number of objectives - 1 ({})."
                   ).format(len(self.epsilons), len(self.objectives(xs)) - 1)
            raise ECMError(msg)

        # evaluate values of epsilon constraint functions
        e: np.ndarray = np.array([-(f - v) for f, v in zip(epsilon_left_side, self.epsilons)])

        if self.constraints(xs) is not None:
            c = self.constraints(xs)
            return np.concatenate([c, e], axis=None)  # does it work with multiple constraints?
        else:
            return e

    def __call__(self, objective_vector: np.ndarray) -> Union[float, np.ndarray]:
        """
        Returns the value of objective function to be minimized.

        Args:
            objective_vector (np.ndarray): Values of objective functions.

        Returns:
            Value of objective function to be minimized.
        """
        if np.shape(objective_vector)[0] > 1:  # more rows than one
            return np.array([objective_vector[i][self._to_be_minimized] for i, _ in enumerate(objective_vector)])
        else:
            return objective_vector[0][self._to_be_minimized]


# Testing the method
if __name__ == "__main__":
    # 1. Define objective functions, bounds and constraints

    def volume(r, h):
        return np.pi * r ** 2 * h

    def area(r, h):
        return 2 * np.pi ** 2 + np.pi * r * h

    # add third objective

    def weight(v):
        return 0.01 * v

    def objective(xs):
        # xs is a 2d array like, which has different values for r and h on its first and second columns respectively.
        xs = np.atleast_2d(xs)
        return np.stack((volume(xs[:, 0], xs[:, 1]), -area(xs[:, 0], xs[:, 1]), weight(volume(xs[:, 0], xs[:, 1])))).T

    # bounds for decision variables
    r_bounds = np.array([2.5, 15])
    h_bounds = np.array([10, 50])
    bounds = np.stack((r_bounds, h_bounds))

    # constraints

    def con_golden(xs):
        # constraints are defined in DESDEO in a way were a positive value indicates an agreement with a constraint, and
        # a negative one a disagreement.
        xs = np.atleast_2d(xs)
        return -(xs[:, 0] / xs[:, 1] - 1.618)

    # 2. Apply Epsilon contraint method

    # index of which objective function to minimize
    obj_min = 2

    # set upper bound(s) for the other objectives, in the same order than which corresponding objective functions
    # are defined
    epsil = np.array([2000, -100])  # multiply the epsilons with -1, if the constraint is of form f_i(x) >= e_i

    # create an instance of EpsilonConstraintMethod-class for given problem
    eps = EpsilonConstraintMethod(objective, obj_min, epsil, constraints=con_golden)

    # constraint evaluator, used by the solver
    cons_evaluate = eps.evaluate_constraints

    # scalarize
    scalarized_objective = Scalarizer(objective, eps)
    print(scalarized_objective)

    # 3. Solve

    # starting point
    x0 = np.array([2, 11])
    minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=cons_evaluate, method=None)

    # minimize
    res = minimizer.minimize(x0)
    final_r, final_h = res["x"][0], res["x"][1]
    final_obj = objective(res["x"]).squeeze()
    final_V, final_A, final_W = final_obj[0], final_obj[1], final_obj[2]

    print(f"Final cake specs: radius: {final_r}cm, height: {final_h}cm.")
    print(f"Final cake dimensions: volume: {final_V}, area: {-final_A}, weight: {final_W}.")
    print(final_r / final_h)
    print(res)
