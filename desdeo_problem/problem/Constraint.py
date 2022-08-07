"""Module for problem constraint definition related classes and functions.

"""
from abc import ABC, abstractmethod
from os import path
from typing import Callable, List

import numpy as np


class ConstraintError(Exception):
    """Raised when an error related to the Constraint class in encountered."""


class ConstraintBase(ABC):
    """Base class for constraints."""

    @abstractmethod
    def evaluate(self, decision_vector: np.ndarray, objective_vector: np.ndarray) -> float:
        """Evaluate the constraint functions.

        This function will evaluate constraints and return a float
        indicating how severely the constraint has been broken.

        Arguments:
            decision_vector (np.ndarray): A decision_vector containing
                the decision variable values.
            objective_vector (np.ndarray): A decision_vector containing the
                objective function values.

        Returns:
            float: A float representing how and if the constraint has
            been violated. A positive value represents no violation and a
            negative value represents a violation. The absolute value of the
            returned float functions as an indicator of the severity of the
            violation (or how well the constraint holds, if the returned value
            of positive).

        """
        pass


class ScalarConstraint(ConstraintBase):
    """A simple scalar constraint that evaluates to a single scalar.

    Arguments:
        name (str): Name of the constraint.
        n_decision_vars (int): Number of decision variables present in the
            constraint.
        n_objective_funs (int): Number of objective functions present in
            the constraint.
        evaluator (Callable): A callable to evaluate the constraint.

    Attributes:
        __name (str): Name of the constraint.
        __n_decision_vars (int): Number of decision variables present in the
            constraint.
        __n_objective_funs (int): Number of objective functions present in
            the constraint.
        __evaluator (Callable): A callable to evaluate the constraint.

    """

    def __init__(
        self,
        name: str,
        n_decision_vars: int,
        n_objective_funs: int,
        evaluator: Callable,
    ) -> None:
        self.__name: str = name
        self.__n_decision_vars: int = n_decision_vars
        self.__n_objective_funs: int = n_objective_funs
        self.__evaluator: Callable = evaluator

    @property
    def name(self) -> str:
        """Property: name

        Returns:
            str: Name of the constraint
        """
        return self.__name

    @property
    def n_decision_vars(self) -> int:
        """Property: number of decision variables

        Returns:
            int: Number of decision variables
        """
        return self.__n_decision_vars

    @property
    def n_objective_funs(self) -> int:
        """Property: number of objective functions

        Returns:
            int: Number of objective functions
        """
        return self.__n_objective_funs

    @property
    def evaluator(self) -> Callable:
        """Property: constraint evaluator callable

        Returns:
            Callable: A callable to evaluate the constraint.
        """
        return self.__evaluator

    def evaluate(self, decision_vector: np.ndarray, objective_vector: np.ndarray) -> float:
        """Evaluate the constraint.

        This evaluates the constraint and return a float indicating how and if the
        constraint was violated. A negative value indicates a violation and
        a positive value indicates a non-violation.

        Arguments:
            decision_vector (np.ndarray): A decision_vector containing the
                values of the decision variables.
            objective_vector (np.ndarray): A decision_vector containing the
                values of the objective functions.

        Returns:
            float: A float indicating how the constraint holds.

        Raises:
            ConstraintError: When something goes wrong evaluating the
                constraint or the objectives and decision vectors are of wrong
                shape.

        """
        if not isinstance(decision_vector, np.ndarray):
            raise ConstraintError("Decision vector needs to be numpy array")

        if not isinstance(objective_vector, np.ndarray):
            raise ConstraintError("Objective vector needs to be numpy array")

        decision_l = len(decision_vector) if decision_vector.ndim == 1 else decision_vector.shape[1]
        if decision_l != self.__n_decision_vars:
            msg = ("Decision decision_vector {} is of wrong lenght: " "Should be {}, but is {}").format(
                decision_vector, self.__n_decision_vars, decision_l
            )
            raise ConstraintError(msg)

        objective_l = len(objective_vector) if objective_vector.ndim == 1 else objective_vector.shape[1]
        if objective_l != self.__n_objective_funs:
            msg = ("Objective decision_vector {} is of wrong lenght:" " Should be {}, but is {}").format(
                objective_vector, self.__n_objective_funs, objective_l
            )
            raise ConstraintError(msg)
        try:
            result = self.__evaluator(decision_vector, objective_vector)
        except (TypeError, IndexError) as e:
            msg = ("Bad arguments {} and {} supplied to the evaluator:" " {}").format(
                str(decision_vector), objective_vector, str(e)
            )
            raise ConstraintError(msg)

        return result


supported_operators: List[str] = ["==", "<", ">"]
"""List[str]: Shows the operators supported in the
constraint_function_factory."""


def constraint_function_factory(lhs: Callable, rhs: float, operator: str) -> Callable:
    """A function that creates an evaluator.

    This function creates an evaluator to be used with the ScalarConstraint
    class. Constraints should be formulated in a way where all the mathematical
    expression are on the left hand side, and the constants on the right hand
    side.

    Arguments:
        lhs (Callable): The left hand side of the constraint. Should be a
            callable function representing a mathematical expression.
        rhs (float): The right hand side of a constraint. Represents the right
            hand side of the constraint.
        operator (str): The kind of constraint. Can be '==', '<', '>'.

    Returns:
        Callable: A function that can be called to evaluate the rhs and
        which returns representing how the constraint is obeyed. A negative
        value represent a violation of the constraint and a positive value an
        agreement with the constraint. The absolute value of the float is a
        direct indicator how the constraint is violated/agdreed with.

    Raises:
        ValueError: The supplied operator is not supported.

    """
    if operator not in supported_operators:
        msg = "The operator {} supplied is not supported.".format(operator)
        raise ValueError(msg)

    if operator == "==":

        def equals(decision_vector: np.ndarray, objective_vector: np.ndarray) -> float:
            return -abs(lhs(decision_vector, objective_vector) - rhs)

        return equals

    elif operator == "<":

        def lt(decision_vector: np.ndarray, objective_vector: np.ndarray) -> float:
            return rhs - lhs(decision_vector, objective_vector)

        return lt

    elif operator == ">":

        def gt(decision_vector: np.ndarray, objective_vector: np.ndarray) -> float:
            return lhs(decision_vector, objective_vector) - rhs

        return gt

    else:
        # if for some reason a bad operator falls through
        msg = "Bad operator argument supplied: {}".format(operator)
        raise ValueError(msg)
