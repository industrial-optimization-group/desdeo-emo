"""Desdeo-problem related definitions.

This file has following classes:
    ProblemError
    EvaluationResults
    ProblemBase
    ScalarMOProblem  -- to be deprecated
    ScalarDataProblem -- to be deprecated
    MOProblem
    DataProblem
    ExperimentalProblem
    IOPISProblem
    DiscreteDataProblem
"""
from abc import ABC, abstractmethod

# , TypedDict coming in py3.8
from functools import reduce
from operator import iadd
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd

from desdeo_problem.problem.Constraint import ScalarConstraint
from desdeo_problem.problem.Objective import (
    VectorDataObjective,
    VectorObjective,
    _ScalarDataObjective,
    _ScalarObjective,
    ScalarObjective,
    ScalarDataObjective,
)
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor
from desdeo_problem.problem.Variable import Variable


class ProblemError(Exception):
    """Raised when an error related to the Problem class is encountered.

    """


# TODO consider replacing namedtuple with attr.s for validation purposes.


class EvaluationResults(NamedTuple):
    """The return object of <problem>.evaluate methods.

    Attributes:
        objectives (np.ndarray): The objective function values for each input
            vector.
        fitness (np.ndarray): Equal to objective values if objective is to be
            minimized. Multiplied by (-1) if objective to be maximized.
        constraints (Union[None, np.ndarray]): The constraint values of the
            problem corresponding each input vector.
        uncertainity (Union[None, np.ndarray]): The uncertainity in the
            objective values.

    """

    objectives: np.ndarray
    fitness: np.ndarray
    constraints: Union[None, np.ndarray] = None
    uncertainity: Union[None, np.ndarray] = None

    def __str__(self):
        """Textual output of attributes.

        Returns:
            str: The textual output of attributes
        """
        prnt_msg = (
            "Evaluation Results Object \n"
            f"Objective values are: \n{self.objectives}\n"
            f"Constraint violation values are: \n{self.constraints}\n"
            f"Fitness values are: \n{self.fitness}\n"
            f"Uncertainity values are: \n{self.uncertainity}\n"
        )
        return prnt_msg


class ProblemBase(ABC):
    """The base class for the problems.

    All other problem classes should be derived from this.

    Attributes:
        nadir (np.ndarray): Nadir values for the problem, initiated = None
        ideal (np.ndarray): Ideal values for the problem, initiated = None
        nadir_fitness (np.ndarray): Fitness values for nadir, initiated = None
        ideal_fitness (np.ndarray): Fitness values for ideal, initiated = None
        __n_of_objectives (int): Number of objectives, initiated = 0
        __n_of_variables (int): Number of variables, initiated = 0
        __decision_vectors (np.ndarray): Array of decision variable vectors,
                                            initiated = None
        __objective_vectors (np.ndarray): Array of objective variable vectors,
                                            initiated = None
    """

    def __init__(self):
        self.nadir: np.ndarray = None
        self.ideal: np.ndarray = None
        self.nadir_fitness: np.ndarray = None
        self.ideal_fitness: np.ndarray = None
        self.__n_of_objectives: int = 0
        self.__n_of_variables: int = 0
        self.__decision_vectors: np.ndarray = None
        self.__objective_vectors: np.ndarray = None

    @property
    def n_of_objectives(self) -> int:
        """Property, returns the number of objectives.

        Returns:
            int: The number of objectives
        """
        return self.__n_of_objectives

    @n_of_objectives.setter
    def n_of_objectives(self, val: int):
        """Setter, the number of objectives.

        Arguments:
            val (int): the number of objectives
        """
        self.__n_of_objectives = val

    @property
    def n_of_variables(self) -> int:
        """Property, returns the number of variables.

        Returns:
            int: The number of variables
        """
        return self.__n_of_variables

    @n_of_variables.setter
    def n_of_variables(self, val: int):
        """Setter, the number of variables.

        Arguments:
            val (int): the number of variables
        """
        self.__n_of_variables = val

    @property
    def decision_vectors(self) -> np.ndarray:
        """Property, returns the decision variable vectors array.

        Returns:
            np.ndarray: decision vector array
        """
        return self.__decision_vectors

    @decision_vectors.setter
    def decision_vectors(self, val: np.ndarray):
        """Setter, the decision variable vector array.

        Arguments:
            val (np.ndarray): the decision vector array
        """
        self.__decision_vectors = val

    @abstractmethod
    def get_variable_bounds(self) -> Union[None, np.ndarray]:
        """Abstract method to get variable bounds"""
        pass

    @abstractmethod
    def evaluate(
        self, decision_vectors: np.ndarray, use_surrogate: bool = False
    ) -> EvaluationResults:
        """Abstract method to evaluate problem.
                                                                   
        Evaluates the problem using an ensemble of input vectors. Uses
        surrogate models if available. Otherwise, it uses the true evaluator.

        Arguments:
            decision_vectors (np.ndarray): An array of decision variable
            input vectors.
            use_surrogate (bool): A bool to control whether to use the true, potentially
            expensive function or a surrogate model to evaluate the objectives.

        Returns:
            (Dict): Dict with the following keys:
                'objectives' (np.ndarray): The objective function values for each input
                    vector.
                'constraints' (Union[np.ndarray, None]): The constraint values of the
                    problem corresponding each input vector.
                'fitness' (np.ndarray): Equal to objective values if objective is to be
                    minimized. Multiplied by (-1) if objective to be maximized.
                'uncertainity' (Union[np.ndarray, None]): The uncertainity in the
                    objective values.

        """

    @abstractmethod
    def evaluate_constraint_values(self) -> Optional[np.ndarray]:
        """Abstract method to evaluate constraint values.

        Evaluate just the constraint function values using the attributes
        decision_vectors and objective_vectors
        
        Note:
            Currently not supported by ScalarMOProblem

        """


# TODO: Depreciate. Use MO problem in the future
class ScalarMOProblem(ProblemBase):
    """A multiobjective optimization problem.

    To be depreciated.

    A multiobjective optimization problem with user defined objective
    functions, constraints and variables.
    The objectives each return a single scalar.

    Arguments:   
        objectives (List[ScalarObjective]): A list containing the objectives of
            the problem.
        variables (List[Variable]): A list containing the variables of the
            problem.
        constraints (List[ScalarConstraint]): A list containing the
            constraints of the problem. If no constraints exist, None may
            be supllied as the value.
        nadir (Optional[np.ndarray]): The nadir point of the problem.
        ideal (Optional[np.ndarray]): The ideal point of the problem.

    Attributes:
        __n_of_objectives (int): The number of objectives in the problem.
        __n_of_variables (int): The number of variables in the problem.
        __n_of_constraints (int): The number of constraints in the problem.
        __nadir (np.ndarray): The nadir point of the problem.
        __ideal (np.ndarray): The ideal point of the problem.
        __objectives (List[ScalarObjective]): A list containing the objectives of
            the problem.
        __constraints (List[ScalarConstraint]): A list conatining the constraints
            of the problem.

    Raises:
        ProblemError: Ill formed nadir and/or ideal vectors are supplied.

    """

    def __init__(
        self,
        objectives: List[ScalarObjective],
        variables: List[Variable],
        constraints: List[ScalarConstraint],
        nadir: Optional[np.ndarray] = None,
        ideal: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.__objectives: List[ScalarObjective] = objectives
        self.__variables: List[Variable] = variables
        self.__constraints: List[ScalarConstraint] = constraints

        self.__n_of_objectives: int = len(self.objectives)
        self.__n_of_variables: int = len(self.variables)

        if self.constraints is not None:
            self.__n_of_constraints: int = len(self.constraints)
        else:
            self.__n_of_constraints = 0

        # Nadir vector must be the same size as the number of objectives
        if nadir is not None:
            if len(nadir) != self.n_of_objectives:
                msg = (
                    "The length of the nadir vector does not match the"
                    "number of objectives: Length nadir {}, number of "
                    "objectives {}."
                ).format(len(nadir), self.n_of_objectives)
                raise ProblemError(msg)

        # Ideal vector must be the same size as the number of objectives
        if ideal is not None:
            if len(ideal) != self.n_of_objectives:
                msg = (
                    "The length of the ideal vector does not match the"
                    "number of objectives: Length ideal {}, number of "
                    "objectives {}."
                ).format(len(ideal), self.n_of_objectives)
                raise ProblemError(msg)

        # Nadir and ideal vectors must match in size
        if nadir is not None and ideal is not None:
            if len(nadir) != len(ideal):
                msg = (
                    "The length of the nadir and ideal point don't match:"
                    " length of nadir {}, length of ideal {}."
                ).format(len(nadir), len(ideal))
                raise ProblemError(msg)

        self.__nadir = nadir
        self.__ideal = ideal

        # Multiplier to convert maximization to minimization
        max_multiplier = np.asarray([1, -1])
        to_maximize = [objective.maximize for objective in objectives]
        to_maximize = sum(to_maximize, [])  # To flatten the list
        to_maximize = np.asarray(to_maximize) * 1  # Convert to zeros and ones
        self._max_multiplier = max_multiplier[to_maximize]

    @property
    def n_of_constraints(self) -> int:
        """Property: the number of constraints.

        Returns:
            int: the number of constraints.
        """
        return self.__n_of_constraints

    @n_of_constraints.setter
    def n_of_constraints(self, val: int):
        """Setter: the number of constraints.

        Arguments:
            int: the number of constraints.
        """
        self.__n_of_constraints = val

    @property
    def objectives(self) -> List[ScalarObjective]:
        """Property: the list of objectives.

        Returns:
            List[ScalarObjective]: the list of objectives
        """
        return self.__objectives

    @objectives.setter
    def objectives(self, val: List[ScalarObjective]):
        """Setter: the list of objectives.

        Arguments:
            val (List[ScalarObjective]): the list of objectives.
        """
        self.__objectives = val

    @property
    def variables(self) -> List[Variable]:
        """Property: the list of problem variables.

        Returns:
            List[_ScalarObjective]: the list of problem variables
        """
        return self.__variables

    @variables.setter
    def variables(self, val: List[Variable]):
        """Setter: the list of variables.

        Arguments:
            val (List[_ScalarObjective]): the list of variables.
        """
        self.__variables = val

    @property
    def constraints(self) -> List[ScalarConstraint]:
        """Property: the list of constraints.

        Returns:
            List[_ScalarObjective]: the list of constraints
        """
        return self.__constraints

    @constraints.setter
    def constraints(self, val: List[ScalarConstraint]):
        """Setter: the list of constraints.

        Arguments:
            val (List[_ScalarObjective]): the list of constraints.
        """
        self.__constraints = val

    @property
    def n_of_objectives(self) -> int:
        """Property: the number of objectives.

        Returns:
            int: the number of objectives.
        """
        return self.__n_of_objectives

    @n_of_objectives.setter
    def n_of_objectives(self, val: int):
        """Setter: the number of objectives.

        Arguments:
            int: the number of objectives.
        """
        self.__n_of_objectives = val

    @property
    def n_of_variables(self) -> int:
        """Property: the number of variables.

        Returns:
            int: the number of variables.

        """
        return self.__n_of_variables

    @n_of_variables.setter
    def n_of_variables(self, val: int):
        """Setter: the number of variables.

        Arguments:
            int: the number of variables.
        """
        self.__n_of_variables = val

    @property
    def nadir(self) -> np.ndarray:
        """Property: the nadir point of the problem.

        Returns:
            np.ndarray: the nadir point of the problem.
        """
        return self.__nadir

    @nadir.setter
    def nadir(self, val: np.ndarray):
        """Setter: the nadir point of the problem.

        Arguments:
            val (np.ndarray): The nadir point of the problem.

        """
        self.__nadir = val

    @property
    def ideal(self) -> np.ndarray:
        """Property: the ideal point of the problem.

        Returns:
            np.ndarray: the ideal point of the problem.
        """
        return self.__ideal

    @ideal.setter
    def ideal(self, val: np.ndarray):
        """Setter: the ideal point of the problem.

        Arguments:
            val (np.ndarray): The ideal point of the problem.

        """
        self.__ideal = val

    def get_variable_bounds(self) -> Union[np.ndarray, None]:
        """Get the variable bounds.

        Return the upper and lower bounds of each decision variable present
        in the problem as a 2D numpy array. The first column corresponds to the
        lower bounds of each variable, and the second column to the upper
        bound.

        Returns:
            np.ndarray: Lower and upper bounds of each variable
            as a 2D numpy array. If undefined variables, return None instead.

        """
        if self.variables is not None:
            bounds = np.ndarray((self.n_of_variables, 2))
            for ind, var in enumerate(self.variables):
                bounds[ind] = np.array(var.get_bounds())
            return bounds
        else:
            return None

    def get_variable_names(self) -> List[str]:
        """Get variable names.

        Return the variable names of the variables present in the problem in
        the order they were added.

        Returns:
            List[str]: Names of the variables in the order they were added.

        """
        return [var.name for var in self.variables]

    def get_objective_names(self) -> List[str]:
        """Get objective names.

        Return the names of the objectives present in the problem in the
        order they were added.

        Returns:
            List[str]: Names of the objectives in the order they were added.

        """
        return [obj.name for obj in self.objectives]

    def get_uncertainty_names(self) -> List[str]:
        """Return the names of the objectives present in the problem in the
        order they were added.

        Returns:
            List[str]: Names of the objectives in the order they were added.

        """
        return [unc.name for unc in self.objectives]

    def get_variable_lower_bounds(self) -> np.ndarray:
        """Get variable lower bounds.

        Return the lower bounds of each variable as a list. The order of the bounds
        follows the order the variables were added to the problem.

        Returns:
            np.ndarray: An array with the lower bounds of the variables.
        """
        return np.array([var.get_bounds()[0] for var in self.variables])

    def get_variable_upper_bounds(self) -> np.ndarray:
        """Get variable upper bounds.
                                             
        Return the upper bounds of each variable as a list. The order of the
        bounds follows the order the variables were added to the problem.
        Returns:
            np.ndarray: An array with the upper bounds of the variables.
        """
        return np.array([var.get_bounds()[1] for var in self.variables])

    def evaluate(
        self, decision_vectors: np.ndarray, use_surrogate: bool = False
    ) -> EvaluationResults:
        """Evaluates the problem using an ensemble of input vectors.

        Arguments:
            decision_vectors (np.ndarray): An 2D array of decision variable
                input vectors. Each column represent the values of each decision
                variable.

        Returns:
            Tuple[np.ndarray, Union[None, np.ndarray]]: If constraint are
                defined, returns the objective vector values and corresponding
                constraint values. Or, if no constraints are defined, returns just
                the objective vector values with None as the constraint values.

        Raises:
            ProblemError: The decision_vectors have wrong dimensions.

        """
        # Reshape decision_vectors with single row to work with the code
        if use_surrogate is True:
            raise NotImplementedError(
                "Surrogates not yet supported in this class. "
                "Use the '''DataProblem''' class instead."
            )
        shape = np.shape(decision_vectors)
        if len(shape) == 1:
            decision_vectors = np.reshape(decision_vectors, (1, shape[0]))

        (n_rows, n_cols) = np.shape(decision_vectors)

        if n_cols != self.n_of_variables:
            msg = (
                "The length of the input vectors does not match the number "
                "of variables in the problem: Input vector length {}, "
                "number of variables {}."
            ).format(n_cols, self.n_of_variables)
            raise ProblemError(msg)

        objective_vectors: np.ndarray = np.ndarray(
            (n_rows, self.n_of_objectives), dtype=float
        )  # ??? Use np.zeros instead of this?
        uncertainity: np.ndarray = np.ndarray(
            (n_rows, self.n_of_objectives), dtype=float
        )  # ??? Use np.zeros instead of this?
        if self.n_of_constraints > 0:
            constraint_values: np.ndarray = np.ndarray(
                (n_rows, self.n_of_constraints), dtype=float
            )
        else:
            constraint_values = None

        # Calculate the objective values
        for (col_i, objective) in enumerate(self.objectives):
            results = objective.evaluate(decision_vectors)
            objective_vectors[:, col_i] = results.objectives
            uncertainity[:, col_i] = results.uncertainity
        # Calculate fitness, which is always to be minimized
        fitness = objective_vectors * self._max_multiplier

        # Calculate the constraint values
        if constraint_values is not None:
            for (col_i, constraint) in enumerate(self.constraints):
                constraint_values[:, col_i] = np.array(
                    constraint.evaluate(decision_vectors, objective_vectors)
                )

        return EvaluationResults(
            objective_vectors, fitness, constraint_values, uncertainity
        )

    def evaluate_constraint_values(self) -> Optional[np.ndarray]:
        """Evaluate constraint values.

        Evaluate just the constraint function values using the attributes
        decision_vectors and objective_vectors

        Raises:
            NotImplementedError

        Note:
            Currently not supported by ScalarMOProblem

        """
        raise NotImplementedError("Not implemented for ScalarMOProblem")


# TODO: Depreciate. Use data problem in the future
class ScalarDataProblem(ProblemBase):
    """A problem class for case where the data is pre-computed.

    To be depreciated
    
    Defines a problem with pre-computed data representing a multiobjective
    optimization problem with scalar valued objective functions.

    Arguments:
        decision_vectors (np.ndarray): A 2D vector of decision_vectors. Each
            row represents a solution with the value for each decision_vectors
            defined on the columns.
        objective_vectors (np.ndarray): A 2D vector of
            objective function values. Each row represents one objective vector
            with the values for the invidual objective functions defined on the
            columns.

    Attributes:
        decision_vectors (np.ndarray): See args
        objective_vectors (np.ndarray): See args
        __epsilon (float): A small floating point number to shift the bounds of
            the variables. See, get_variable_bounds, default value 1e-6
        __constraints (List[ScalarConstraint]): A list of defined constraints.
        nadir (np.ndarray): The nadir point of the problem.
        ideal (np.ndarray): The ideal point of the problem.
        __model_exists (bool): is there a model for this problem

    Note:
        It is assumed that the decision_vectors and objectives follow a direct
        one-to-one mapping, i.e., the objective values on the ith row in
        'objectives' should represent the solution of the multiobjective
        problem when evaluated with the decision_vectors on the ith row in
        'decision_vectors'.

    """

    def __init__(self, decision_vectors: np.ndarray, objective_vectors: np.ndarray):
        super().__init__()
        self.decision_vectors: np.ndarray = decision_vectors
        self.objective_vectors: np.ndarray = objective_vectors
        # epsilon is used when computing the bounds. We don't want to exclude
        # any of the solutions that contain border values.
        # See get_variable_bounds
        self.__epsilon: float = 1e-6
        # Used to indicate if a model has been built to represent the model.
        # Used in the evaluation.
        self.__model_exists: bool = False
        self.__constraints: List[ScalarConstraint] = []

        try:
            self.n_of_variables = self.decision_vectors.shape[1]
        except IndexError as e:
            msg = (
                "Check the variable dimensions. Is it a 2D array? "
                "Encountered '{}'".format(str(e))
            )
            raise ProblemError(msg)

        try:
            self.n_of_objectives = self.objective_vectors.shape[1]
        except IndexError as e:
            msg = (
                "Check the objective dimensions. Is it a 2D array? "
                "Encountered '{}'".format(str(e))
            )
            raise ProblemError(msg)

        self.nadir = np.max(self.objective_vectors, axis=0)
        self.ideal = np.min(self.objective_vectors, axis=0)

    @property
    def epsilon(self) -> float:
        """Property: epsilon.

        Return:
            float: epsilon value (for shifting the bounds of variables)
        """
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, val: float):
        """Setter: epsilon.

        Argument:
            val (float): epsilon value (for shifting the bounds of variables.)
        """
        self.__epsilon = val

    @property
    def constraints(self) -> List[ScalarConstraint]:
        """Property: Constraints.

        Return:
            List[ScalarConstraint]: list of the defined constraints
        """
        return self.__constraints

    @constraints.setter
    def constraints(self, val: List[ScalarConstraint]):
        """Setter: Constraints.

        Argument:
            val (List[ScalarConstraint]): list of the constraints.
        """
        self.__constraints = val

    def get_variable_bounds(self):
        """Get the variable bounds.

        Returns:
            np.array[float]: The variable bounds in a stack. The epsilon value
                will be added to the upper bounds and substracted from the
                lower bounds to return closed bounds.

        Note:
            If self.epsilon is zero, the bounds will represent an open range.

        """
        return np.stack(
            (
                np.min(self.decision_vectors, axis=0) - self.epsilon,
                np.max(self.decision_vectors, axis=0) + self.epsilon,
            ),
            axis=1,
        )

    def evaluate_constraint_values(self) -> Optional[np.ndarray]:
        """Evaluate the constraint values.
        
        Evaluate the constraint values for each defined constraint. A positive value indicates that a constraint is adhered to, a negative
        value indicates a violated constraint.

        Returns:
            Optional[np.ndarray]: A 2D array with each row representing the
                constraint values for different objective vectors. One column for
                each constraint. If no constraint function are defined, returns
                None.

        """
        if len(self.constraints) == 0:
            return None

        constraint_values = np.zeros(
            (len(self.objective_vectors), len(self.constraints))
        )

        for ind, con in enumerate(self.constraints):
            constraint_values[:, ind] = con.evaluate(
                self.decision_vectors, self.objective_vectors
            )

        return constraint_values

    def evaluate(self, decision_vectors: np.ndarray) -> np.ndarray:
        """Evaluate the values of the objectives at the given decision.

        Evaluate the values of the objectives corresponding to the decision
        decision_vectors.

        Args:
            decision_vectors (np.ndarray): A 2D array with the decision
                decision_vectors to be evaluated on each row.

        Returns:
            nd.ndarray: A 2D array with the objective values corresponding to
                each decision vectors on the rows.

        Note:
            At the moment, this function just maps the given decision
            decision_vectors to the closest decision variable present (using an
            L2 distance) in the problem and returns the corresponsing objective
            vector.

        """
        if not self.__model_exists:
            idx = np.unravel_index(
                np.linalg.norm(
                    self.decision_vectors - decision_vectors, axis=1
                ).argmin(),
                self.objective_vectors.shape,
                order="F",
            )[0]

        else:
            msg = "Models not implemented yet for data based problems."
            raise NotImplementedError(msg)

        return (self.objective_vectors[idx],)


class MOProblem(ProblemBase):
    """An user defined multiobjective optimization problem.

    A multiobjective optimization problem with user defined objective
    functions, constraints, and variables.

    Arguments:
        objectives (List[Union[ScalarObjective, VectorObjective]]): A list
                        containing the objectives of the problem.
        variables (List[Variable]): A list containing the variables of
                        the problem.
        constraints (List[ScalarConstraint]): A list of the constraints
                        of the problem.
        nadir (Optional[np.ndarray], optional): Nadir point of the problem.
                                                Defaults to None.
        ideal (Optional[np.ndarray], optional): Ideal point of the problem.
                                                Defaults to None.
    Attributes:
        __objectives (List[Union[ScalarObjective, VectorObjective]]): A list
                        containing the objectives of the problem.
        __variables (List[Variable]): A list containing the variables of
                        the problem.
        __constraints (List[ScalarConstraint]): A list of the constraints
                        of the problem.
        __nadir (Optional[np.ndarray], optional): Nadir point of the problem.
                                                Defaults to None.
        __ideal (Optional[np.ndarray], optional): Ideal point of the problem.
                                                Defaults to None.
        __n_of_variables (int): The number of variables
        __n_of_objectives (int): The number of objectives

    Raises:
        ProblemError: If ideal or nadir vectors are not the same size as
                number of objectives.

    """

    # TODO: use_surrogate : Union[bool, List[bool]]
    def __init__(
        self,
        objectives: List[Union[ScalarObjective, VectorObjective]],
        variables: List[Variable],
        constraints: List[ScalarConstraint] = None,
        nadir: Optional[np.ndarray] = None,
        ideal: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.__objectives: List[Union[ScalarObjective, VectorObjective]] = objectives
        self.__variables: List[Variable] = variables
        self.__constraints: List[ScalarConstraint] = constraints
        self.__n_of_variables: int = len(self.variables)
        self.__n_of_objectives: int = sum(
            map(self.number_of_objectives, self.__objectives)
        )
        if self.constraints is not None:
            self.__n_of_constraints: int = len(self.constraints)
        else:
            self.__n_of_constraints = 0

        # Multiplier to convert maximization to minimization
        max_multiplier = np.asarray([1, -1])
        to_maximize = [objective.maximize for objective in objectives]
        # Does not work
        # to_maximize = sum(to_maximize, [])  # To flatten the list
        to_maximize = (
            np.hstack(to_maximize) * 1
        )  # To flatten list and convert to zeros and ones
        # to_maximize = np.asarray(to_maximize) * 1  # Convert to zeros and ones
        self._max_multiplier = max_multiplier[to_maximize]

        self.nadir_fitness = np.full(self.__n_of_objectives, np.inf, dtype=float)
        self.nadir = self.nadir_fitness * self._max_multiplier
        self.ideal_fitness = np.full(self.__n_of_objectives, np.inf, dtype=float)
        self.ideal = self.ideal_fitness * self._max_multiplier

        # Nadir vector must be the same size as the number of objectives
        if nadir is not None:
            if len(nadir) != self.n_of_objectives:
                msg = (
                    "The length of the nadir vector does not match the"
                    "number of objectives: Length nadir {}, number of "
                    "objectives {}."
                ).format(len(nadir), self.n_of_objectives)
                raise ProblemError(msg)
            self.nadir = nadir

        # Ideal vector must be the same size as the number of objectives
        if ideal is not None:
            if len(ideal) != self.n_of_objectives:
                msg = (
                    "The length of the ideal vector does not match the"
                    "number of objectives: Length ideal {}, number of "
                    "objectives {}."
                ).format(len(ideal), self.n_of_objectives)
                raise ProblemError(msg)
            self.ideal = ideal

        self.nadir_fitness = self.nadir * self._max_multiplier
        self.ideal_fitness = self.ideal * self._max_multiplier

        # Objective and variable names
        self.objective_names = self.get_objective_names()
        self.variable_names = self.get_variable_names()
        self.fitness_names = self.objective_names

    @property
    def n_of_constraints(self) -> int:
        """Property: number of constraints.

        Returns:
            int: Number of constraints
        """
        return self.__n_of_constraints

    @n_of_constraints.setter
    def n_of_constraints(self, val: int):
        """Setter: number of constraints.

        Arguments:
            val (int): number of constraints
        """
        self.__n_of_constraints = val

    @property
    def objectives(self) -> List[ScalarObjective]:
        """Property: list of objectives.

        Returns:
            List[ScalarObjective]: list of objectives
        """
        return self.__objectives

    @objectives.setter
    def objectives(self, val: List[ScalarObjective]):
        """Setter: set list of objectives.

        Arguments:
            val (List[ScalarObjective]): List of objectives
        """
        self.__objectives = val

    @property
    def variables(self) -> List[Variable]:
        """Property: List of variables

        Returns:
            List[Variable]: list of variables
        """
        return self.__variables

    @variables.setter
    def variables(self, val: List[Variable]):
        """Setter: set list of variables.

        Arguments:
            val (List[Variable]): list of variables
        """
        self.__variables = val

    @property
    def constraints(self) -> List[ScalarConstraint]:
        """Property: list of constraints.

        Returns:
            List[ScalarConstraint]: list of constraints
        """
        return self.__constraints

    @constraints.setter
    def constraints(self, val: List[ScalarConstraint]):
        """Setter: list of constraints.

        Arguments:
            val (List[ScalarConstraint]): list of constraints
        """
        self.__constraints = val

    @property
    def n_of_objectives(self) -> int:
        """Property: number of objectives.

        Returns:
            int: number of objectives
        """
        return self.__n_of_objectives

    @n_of_objectives.setter
    def n_of_objectives(self, val: int):
        """Setter: number of objectives.

        Arguments:
            val (int): number of objectives
        """
        self.__n_of_objectives = val

    @property
    def n_of_fitnesses(self) -> int:
        """Property: number of dimensions of the fitness matrix.
        May be different than the number of objectives in inherited classes.

        Returns:
            int: number of fitness dimensions.
        """
        return self.__n_of_objectives

    @property
    def n_of_variables(self) -> int:
        """Property: number of variables.

        Returns:
            int: Number of variables.
        """
        return self.__n_of_variables

    @n_of_variables.setter
    def n_of_variables(self, val: int):
        """Setter: number of variables.

        Arguments:
            val (int): number of variables
        """
        self.__n_of_variables = val

    @staticmethod
    def number_of_objectives(
        obj_instance: Union[ScalarObjective, VectorObjective]
    ) -> int:
        """Return the number of objectives in the given obj_instance.

        Arguments:
            obj_instance (Union[ScalarObjective, VectorObjective]): An instance of one of
                the objective classes

        Raises:
            ProblemError: Raised when obj_instance is not an instance of the supported
                classes

        Returns:
            int: Number of objectives in obj_instance
        """
        if isinstance(obj_instance, _ScalarObjective):
            return 1
        elif isinstance(obj_instance, ScalarObjective):
            return 1
        elif isinstance(obj_instance, VectorObjective):
            return obj_instance.n_of_objectives
        else:
            msg = "Supported objective types: ScalarObjective and VectorObjective"
            raise ProblemError(msg)

    def get_variable_bounds(self) -> Union[np.ndarray, None]:
        """Get variable bounds.
        Return the upper and lower bounds of each decision variable present
        in the problem as a 2D numpy array. The first column corresponds to the
        lower bounds of each variable, and the second column to the upper
        bound.

        Returns:
           np.ndarray: Lower and upper bounds of each variable
           as a 2D numpy array. If undefined variables, return None instead.

        """
        if self.variables is not None:
            bounds = np.ndarray((self.n_of_variables, 2))
            for ind, var in enumerate(self.variables):
                bounds[ind] = np.array(var.get_bounds())
            return bounds
        else:
            return None

    def get_variable_names(self) -> List[str]:
        """Get variable names.     
        Return the variable names of the variables present in the problem in
        the order they were added.

        Returns:
            List[str]: Names of the variables in the order they were added.

        """
        return [var.name for var in self.variables]

    def get_objective_names(self) -> List[str]:
        """Get objective names.   
        Return the names of the objectives present in the problem in the
        order they were added.

        Returns:
            List[str]: Names of the objectives in the order they were added.

        """
        obj_list = [[(obj.name)] for obj in self.objectives]
        return reduce(iadd, obj_list, [])

    # TODO: add  get_uncertainty_names() for uncertainty values

    def get_variable_lower_bounds(self) -> np.ndarray:
        """Get variable lower bounds.
        
        Return the lower bounds of each variable as a list. The order of the bounds
        follows the order the variables were added to the problem.

        Returns:
            np.ndarray: An array with the lower bounds of the variables.
        """
        return np.array([var.get_bounds()[0] for var in self.variables])

    def get_variable_upper_bounds(self) -> np.ndarray:
        """Get variable upper bounds.

        Return the upper bounds of each variable as a list. The order of the bounds
        follows the order the variables were added to the problem.

        Returns:
            np.ndarray: An array with the upper bounds of the variables.
        """
        return np.array([var.get_bounds()[1] for var in self.variables])

    def evaluate(
        self, decision_vectors: np.ndarray, use_surrogate: bool = False
    ) -> EvaluationResults:
        """Evaluates the problem using an ensemble of input vectors.

        Arguments:
            decision_vectors (np.ndarray): An 2D array of decision variable
                input vectors. Each column represent the values of each decision
                variable.
            use_surrogate (bool): A bool to control whether to use the true, potentially
                expensive function or a surrogate model to evaluate the objectives.

        Returns:
            Tuple[np.ndarray, Union[None, np.ndarray]]: If constraint are
            defined, returns the objective vector values and corresponding
            constraint values. Or, if no constraints are defined, returns just
            the objective vector values with None as the constraint values.

        Raises:
            ProblemError: The decision_vectors have wrong dimensions.
            ValueError: If decision_vectors violate the lower or upper bounds.

        """
        # Reshape decision_vectors with single row to work with the code
        shape = np.shape(decision_vectors)
        if len(shape) == 1:
            decision_vectors = np.reshape(decision_vectors, (1, shape[0]))

        # Checking bounds; warn if bounds are breached.
        if np.any(self.get_variable_lower_bounds() > decision_vectors):
            warn("Some decision variable values violate lower bounds")
        if np.any(self.get_variable_upper_bounds() < decision_vectors):
            warn("Some decision variable values violate upper bounds")

        (n_rows, n_cols) = np.shape(decision_vectors)

        if n_cols != self.n_of_variables:
            msg = (
                "The length of the input vectors does not match the number "
                "of variables in the problem: Input vector length {}, "
                "number of variables {}."
            ).format(n_cols, self.n_of_variables)
            raise ProblemError(msg)

        objective_vectors, uncertainity = self.evaluate_objectives(
            decision_vectors, use_surrogate=use_surrogate
        )

        constraint_values = self.evaluate_constraint_values(
            decision_vectors, objective_vectors
        )

        # Calculate fitness, which is always to be minimized
        fitness = self.evaluate_fitness(objective_vectors)

        # Update ideal values
        self.update_ideal(objective_vectors, fitness)

        return EvaluationResults(
            objective_vectors, fitness, constraint_values, uncertainity
        )

    def evaluate_objectives(
        self, decision_vectors: np.ndarray, use_surrogate: bool = False
    ) -> Tuple[np.ndarray]:
        """Evaluate objective values of the problem

        Arguments:
           decision_vectors (np.ndarray): An 2D array of decision variable
                input vectors. Each column represent the values of each
                decision variable.
           use_surrogate (bool): A bool to control whether to use the true,
                potentially expensive function or a surrogate model to
                evaluate the objectives.

        Returns:
            Tuple[np.ndarray]: Objective vector values with their uncertainty.

        """
        (n_rows, n_cols) = np.shape(decision_vectors)
        objective_vectors: np.ndarray = np.ndarray(
            (n_rows, self.n_of_objectives), dtype=float
        )

        uncertainity: np.ndarray = np.ndarray(
            (n_rows, self.n_of_objectives), dtype=float
        )

        obj_column = 0
        for objective in self.objectives:
            elem_in_curr_obj = self.number_of_objectives(objective)

            if elem_in_curr_obj == 1:
                results = objective.evaluate(decision_vectors, use_surrogate)
                objective_vectors[:, obj_column] = results.objectives
                uncertainity[:, obj_column] = results.uncertainity
            elif elem_in_curr_obj > 1:
                # results = list(map(objective.evaluate, decision_vectors))
                results = objective.evaluate(decision_vectors, use_surrogate)

                objective_vectors[
                    :, obj_column : obj_column + elem_in_curr_obj
                ] = results.objectives

                uncertainity[
                    :, obj_column : obj_column + elem_in_curr_obj
                ] = results.uncertainity

            obj_column = obj_column + elem_in_curr_obj
        return (objective_vectors, uncertainity)

    def evaluate_constraint_values(
        self, decision_vectors: np.ndarray, objective_vectors: np.ndarray
    ) -> Optional[np.ndarray]:
        """Evaluate constraint values

        Evaluate just the constraint function values using the attributes
        decision_vectors and objective_vectors

        Arguments:
            decision_vectors (np.ndarray): An 2D array of decision variable
                input vectors. Each column represent the values of each
                decision variable.
            use_surrogate (bool): A bool to control whether to use the true,
                potentially expensive function or a surrogate model to
                evaluate the objectives.

        Returns:
            Optional[np.ndarray]: if there are constraints, then this returns
                np.ndarray of constraint values, else returns None
        Raises:
            NotImplementedError

        Note:
            Currently not supported by ScalarMOProblem

        """
        if self.n_of_constraints == 0:
            return None
        (n_rows, n_cols) = np.shape(decision_vectors)
        constraint_values: np.ndarray = np.ndarray(
            (n_rows, self.n_of_constraints), dtype=float
        )

        for (col_i, constraint) in enumerate(self.constraints):
            constraint_values[:, col_i] = np.array(
                constraint.evaluate(decision_vectors, objective_vectors)
            )
        return constraint_values

    def evaluate_fitness(self, objective_vectors: np.ndarray) -> np.ndarray:
        """Evaluate fitness of the objectives.

        Arguments:
            objective_vectors (np.ndarray): objective vector array

        Returns:
            np.ndarray: fitness of each objective vector

        """
        return objective_vectors * self._max_multiplier

    def update_ideal(self, objective_vectors: np.ndarray, fitness: np.ndarray):
        """Update the ideal vector

        Arguments:
            objective_vectors (np.ndarray): Objective vectors
            fitness (np.ndarray): fittness of objective vectors.

        """
        self.ideal_fitness = np.amin(np.vstack((self.ideal_fitness, fitness)), axis=0)
        self.ideal = self.ideal_fitness * self._max_multiplier


# TODO: Make this the "main" Problem class?
class DataProblem(MOProblem):
    """A class for a data based problem.

    A problem class for data-based problem. This supports surrogate modelling.
    Data should be given in the form of a pandas dataframe.
    
    Arguments:
        data (pd.DataFrame): The input data. This will be used for training the model.
        variable_names (List[str]): Names of the variables in the dataframe provided.
        objective_names (List[str]): Names of the objectices in the dataframe provided.
        bounds (pd.DataFrame, optional): A pandas DataFrame containing the upper and
            lower bounds of the decision variables. Column names have to be same as
            variable_names. Row names have to be "lower_bound" and "upper_bound".
        objectives (List[Union[ScalarDataObjective,VectorDataObjective,]], optional):
            Objective instances, currently not supported. Defaults to None.
        variables (List[Variable], optional): Variable instances. Defaults to None.
            Currently not supported.
        constraints (List[ScalarConstraint], optional): Constraint instances.
            Defaults to None, which means that there are no constraints.
        nadir (Optional[np.ndarray], optional): Nadir of the problem. Defaults to None.
        ideal (Optional[np.ndarray], optional): Ideal of the problem. Defaults to None.
    Raises:
        ProblemError: When input data is not a dataframe.
        ProblemError: When given objective or variable names are not in dataframe column
        NotImplementedError: When objective instances are passed
        NotImplementedError: When variable instances are passed
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable_names: List[str],
        objective_names: List[str],
        bounds: pd.DataFrame = None,
        maximize: pd.DataFrame = None,
        objectives: List[Union[ScalarDataObjective, VectorDataObjective]] = None,
        variables: List[Variable] = None,
        constraints: List[ScalarConstraint] = None,
        nadir: Optional[np.ndarray] = None,
        ideal: Optional[np.ndarray] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            msg = "Please provide data in the pandas dataframe format"
            raise ProblemError(msg)
        if not all(obj in data.columns for obj in objective_names):
            msg = "Provided objective names not found in provided dataframe columns"
            raise ProblemError(msg)
        if not all(var in data.columns for var in variable_names):
            msg = "Provided variable names not found in provided dataframe columns"
            raise ProblemError(msg)
        if bounds is not None:
            if not all(var in variable_names for var in bounds.columns):
                msg = "A mismatch in the variable names in the bounds dataframe"
                raise ProblemError(msg)
            bounds_row_names = ["lower_bound", "upper_bound"]
            if not all(row_name in bounds_row_names for row_name in bounds.index):
                msg = (
                    f"'bounds' should contain the following indices: {bounds_row_names}"
                )
                raise ProblemError(msg)
        if maximize is not None:
            if not all(obj in objective_names for obj in maximize.columns):
                msg = "maximize DataFrame should only contain objectives"
                raise ProblemError(msg)
            if not all(obj in maximize.columns for obj in objective_names):
                msg = "All objectives should be in the maximize DataFrame"
                raise ProblemError(msg)
        if maximize is None:
            # Default to minimize
            maximize = pd.DataFrame(columns=objective_names, index=[0])
            maximize[:] = False
        # TODO: Implement the rest
        if objectives is not None:
            msg = "Support for custom objectives objects not implemented yet"
            raise NotImplementedError(msg)
        if variables is not None:
            msg = "Support for custom variables objects not implemented yet"
            raise NotImplementedError(msg)
        if objectives is None:
            objectives = []
            for obj in objective_names:
                objectives.append(
                    ScalarDataObjective(
                        data=data[variable_names + [obj]],
                        name=obj,
                        maximize=maximize[obj].tolist(),
                    )
                )
        if variables is None:
            variables = []
            for var in variable_names:
                initial_value = data[var].mean(axis=0)
                if bounds is None:
                    lower_bound = data[var].min(axis=0)
                    upper_bound = data[var].max(axis=0)
                else:
                    lower_bound = bounds[var]["lower_bound"]
                    upper_bound = bounds[var]["upper_bound"]
                variables.append(
                    Variable(
                        name=var,
                        initial_value=initial_value,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                )
        super().__init__(objectives, variables, constraints)

    def train(
        self,
        models: Union[BaseRegressor, List[BaseRegressor]],
        model_parameters: Union[Dict, List[Dict]] = None,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train surrogate models for all the objectives.

        The models should have a fit method and a predict method. The predict method
        should return predicted values as well as uncertainity value (even if they are
        none.)

        Arguments:
            models (Union[BaseRegressor, List[BaseRegressor]]): The class for the
                surrogate modelling algorithm.
            models_parameters: Dict or List[Dict]
                The parameters for the regressors. Should be a dict if a single regressor is
                provided. If a list of regressors is provided, the parameters should be in a
                list of dicts, same length as the list of regressors(= number of objs).
            index (List[int], optional): The indices of the samples to be used for
                training the surrogate model. If no values are proveded, all samples are
                used.
            data (pd.DataFrame, optional): Use this argument if some external data is
                to be used for training. Defaults to None.

        Raises:
            ProblemError: If VectorDataObjective is used as one of the objective
            instances. They are not supported yet.
        """
        if not isinstance(models, list):
            models = [models] * len(self.get_objective_names())
            model_parameters = [model_parameters] * len(self.get_objective_names())
        elif len(models) == 1:
            models = models * len(self.get_objective_names())
        for model, model_params, name in zip(
            models, model_parameters, self.get_objective_names()
        ):
            self.train_one_objective(name, model, model_params, index, data)

    def train_one_objective(
        self,
        name: str,
        model: BaseRegressor,
        model_parameters: Dict,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train one objective at a time, otherwise same is the train method.

        Arguments:
            name (str): Name of the objective to be trained.
            model (BaseRegressor): The class for the surrogate modelling algorithm.
            model_parameters (Dict): **model_parameters is passed to the model when
                initialized.
            index (List[int], optional): The indices of the samples to be used for
                training the surrogate model. If no values are proveded, all samples are
                used.
            data (pd.DataFrame, optional): Use this argument if some external data is
                to be used for training. Defaults to None.

        Raises:
            ProblemError: If name is not in the list of objective names.
            ProblemError: If VectorDataObjective is used as one of the objective
                instances. They are not supported yet.
        """
        if name not in self.get_objective_names():
            raise ProblemError(
                f'"{name}" not found in the list of'
                f"original objective names: {self.get_objective_names()}"
            )
        obj_index = self.get_objective_names().index(name)
        if isinstance(self.objectives[obj_index], _ScalarDataObjective):
            self.objectives[obj_index].train(model, model_parameters, index, data)
        if isinstance(self.objectives[obj_index], ScalarDataObjective):
            self.objectives[obj_index].train(model, model_parameters, index, data)
        else:
            msg = "Support for VectorDataObjective not supported yet"
            raise ProblemError(msg)


class ExperimentalProblem(MOProblem):
    """A problem class for data-based problem. This supports surrogate modelling.
    Data should be given in the form of a pandas dataframe.
    self.archive is created to save the true function evaluations and update the
    surrogate models if needed.
    self.archive updates whenever a true function evaluation happens.  
  
    Arguments:
        data (pd.DataFrame): The input data. This will be used for training the model.
        variable_names (List[str]): Names of the variables in the dataframe provided.
        objective_names (List[str]): Names of the objectices in the dataframe provided.
        objectives (List[Union[ScalarDataObjective,VectorDataObjective,]], optional):
            Objective instances, currently not supported. Defaults to None.
        variables (List[Variable], optional): Variable instances. Defaults to None.
            Currently not supported.
        constraints (List[ScalarConstraint], optional): Constraint instances.
            Defaults to None, which means that there are no constraints.
        nadir (Optional[np.ndarray], optional): Nadir of the problem. Defaults to None.
        ideal (Optional[np.ndarray], optional): Ideal of the problem. Defaults to None.
    
    Raises:
        ProblemError: When input data is not a dataframe.
        ProblemError: When given objective or variable names are not in dataframe column
        NotImplementedError: When objective instances are passed
        NotImplementedError: When variable instances are passed

    Note:
        not properly implemented!

    """

    def __init__(
        self,
        variable_names: List[str],
        objective_names: List[str],
        uncertainity_names: List[str],
        evaluators: Union[None, List[Callable]] = None,
        dimensions_data: pd.DataFrame = None,
        data: pd.DataFrame = None,
        objective_functions: List[Tuple[List[str], Callable]] = None,
        constraints: List[Tuple[List[str], Callable]] = None,
    ):
        # TODO: add the archiving here for true evaluations.

        self.uncertainity_names = uncertainity_names  #  will be removed later
        if not isinstance(data, pd.DataFrame):
            msg = "Please provide data in the pandas dataframe format"
            raise ProblemError(msg)
        if not all(obj in data.columns for obj in objective_names):
            msg = "Provided objective names not found in provided dataframe columns"
            raise ProblemError(msg)
        if not all(var in data.columns for var in variable_names):
            msg = "Provided variable names not found in provided dataframe columns"
            raise ProblemError(msg)
        # TODO: Implement the rest
        objectives = []
        self.archive = (
            data
        )  # this is for model management to archive the solutions and decision variables
        # check if evaluator is NOne in that case make a list of nones and the lenght of the list is the same as obj_names
        # check if evaluator is the same lenght as obj_names if not rais a problem error
        for obj, evaluator in zip(objective_names, evaluators):
            objectives.append(
                ScalarDataObjective(
                    data=data[variable_names + [obj]], name=obj, evaluator=evaluator
                )
            )

        variables = []
        for var in variable_names:
            initial_value = data[var].mean(axis=0)
            lower_bound = data[var].min(axis=0)
            upper_bound = data[var].max(axis=0)
            variables.append(
                Variable(
                    name=var,
                    initial_value=initial_value,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )
        super().__init__(objectives, variables, constraints)

    def train(
        self,
        models: Union[BaseRegressor, List[BaseRegressor]],
        model_parameters: Union[Dict, List[Dict]] = None,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train surrogate models for all the objectives. 
        
        The models should have a fit method and a predict method. The predict 
        method should return predicted values as well as uncertainity value
        (even if they are none.)

        Arguments:
            models (Union[BaseRegressor, List[BaseRegressor]]): The class for the
                surrogate modelling algorithm.
            models_parameters: Dict or List[Dict]
                The parameters for the regressors. Should be a dict if a single regressor is
                provided. If a list of regressors is provided, the parameters should be in a
                list of dicts, same length as the list of regressors(= number of objs).
            index (List[int], optional): The indices of the samples to be used for
                training the surrogate model. If no values are proveded, all samples are
                used.
            data (pd.DataFrame, optional): Use this argument if some external data is
                to be used for training. Defaults to None.

        Raises:
            ProblemError: If VectorDataObjective is used as one of the objective
                instances. They are not supported yet.
        """
        data = self.archive  # for updating the data after updating the surrogates
        if not isinstance(models, list):
            models = [models] * len(self.get_objective_names())
            model_parameters = [model_parameters] * len(self.get_objective_names())
        elif len(models) == 1:
            models = models * len(self.get_objective_names())
        for model, model_params, name in zip(
            models, model_parameters, self.get_objective_names()
        ):
            self.train_one_objective(name, model, model_params, index, data)

    def train_one_objective(
        self,
        name: str,
        model: BaseRegressor,
        model_parameters: Dict,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train one objective at a time, otherwise same is the train method.

        Arguments:
            name (str): Name of the objective to be trained.
            model (BaseRegressor): The class for the surrogate modelling algorithm.
            model_parameters (Dict): **model_parameters is passed to the model when
                initialized.
            index (List[int], optional): The indices of the samples to be used for
                training the surrogate model. If no values are proveded, all samples are
                used.
            data (pd.DataFrame, optional): Use this argument if some external data is
                to be used for training. Defaults to None.

        Raises:
            ProblemError: If name is not in the list of objective names.
            ProblemError: If VectorDataObjective is used as one of the objective
                instances. They are not supported yet.
        """
        if name not in self.get_objective_names():
            raise ProblemError(
                f'"{name}" not found in the list of'
                f"original objective names: {self.get_objective_names()}"
            )
        obj_index = self.get_objective_names().index(name)
        if isinstance(self.objectives[obj_index], _ScalarDataObjective):
            self.objectives[obj_index].train(model, model_parameters, index, data)
        elif isinstance(self.objectives[obj_index], ScalarDataObjective):
            self.objectives[obj_index].train(model, model_parameters, index, data)
        else:
            msg = "Support for VectorDataObjective not supported yet"
            raise ProblemError(msg)

    def evaluate(
        self, decision_vectors: np.ndarray, use_surrogate: bool
    ) -> EvaluationResults:

        names = np.hstack((self.variable_names, self.objective_names))
        new_results = super().evaluate(decision_vectors, use_surrogate=use_surrogate)

        # The following is for achiving solutions with true function evaluations
        if use_surrogate == False:
            new_samples = np.hstack((decision_vectors, new_results.objectives))
            new_data = pd.DataFrame(data=new_samples, columns=names)
            self.archive = self.archive.append(new_data)
        return new_results


class DiscreteDataProblem:
    """A problem class for data-based problems with discrete values.

    These data values are computed representing a set of non-dominated points.

    Arguments:
        data (pd.DataFrame): The input data.
        variable_names (List[str]): Names of the variables in the dataframe provided.
        objective_names (List[str]): Names of the objectices in the dataframe provided.
        nadir (np.ndarray): Nadir of the problem.
        ideal (np.ndarray): Ideal of the problem.
    
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable_names: List[str],
        objective_names: List[str],
        ideal: np.ndarray,
        nadir: np.ndarray,
    ):
        self.decision_variables = data[variable_names].values
        self.variable_names = variable_names
        self.objectives = data[objective_names].values
        self.objective_names = objective_names
        self.ideal = ideal
        self.nadir = nadir
        self.n_of_objectives = len(objective_names)

    def find_closest(self, x: np.ndarray) -> int:
        """Find closest point in data to x.

        Given a vector of decision variables, finds the closest point in
        the given data and returns its index.
        A simple euclidean distance is used.

        Arguments:
            x (np.ndarray): A 1D vector containing decision variables.
        
        Returns:
            int: The index of the closest point in the data computed for x.
        """
        return np.argmin(np.linalg.norm(x - self.decision_variables, axis=1))
