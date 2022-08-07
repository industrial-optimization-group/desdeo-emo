import numpy as np
import pytest
from desdeo_problem.problem.Constraint import ConstraintError, ScalarConstraint, constraint_function_factory


## UTILS

def evaluator(x, y):
    res = np.sum(x) + np.sum(y)
    return float(res)


# tries to get unaccessable indexes
def bad_evaluator(x, y):
    res = x[5] + y[7]
    return float(res)


# TODO: make fixtures of the repetitive params
# Example of testing that ConstraintError works or custom errors in general
def decision_dims_wrong():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    # 2 instead of needed 3
    decision_vector = np.array([1.0, 1.0])
    objective_vector = np.array([1.0, 1.0, 1.0])
    s_const.evaluate(decision_vector, objective_vector)


def objective_dims_wrong():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    # 2 instead of needed 3
    decision_vector = np.array([1.0, 1.0, 1.0])
    objective_vector = np.array([1.0, 1.0])
    s_const.evaluate(decision_vector, objective_vector)


def bad_args():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    # obj vector is wrong type
    decision_vector = np.array([1.0, 1.0, 1.0])
    # objective_vector = np.array([1, 1, 1])
    objective_vector = [1.0, 1.0, 1.0]
    s_const.evaluate(decision_vector, objective_vector)


def bad_eval():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=bad_evaluator)
    # obj vector is wrong type
    decision_vector = np.array([1.0, 1.0, 1.0])
    objective_vector = np.array([1., 1., 1.])
    # objective_vector = np.array([1, 1, 1])
    s_const.evaluate(decision_vector, objective_vector)


# TESTS
def test_decision_dims_fails():
    with pytest.raises(ConstraintError):
        decision_dims_wrong()


def test_obj_dims_fails():
    with pytest.raises(ConstraintError):
        objective_dims_wrong()


# AttributeError test that is not included in Constraint.py
def test_bad_args_fails():
    with pytest.raises(ConstraintError):
        bad_args()


# AttributeError test that is not included in Constraint.py
def test_eval_fails():
    with pytest.raises(ConstraintError):
        bad_eval()


def test_ScalarConstraint():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    decision_vector = np.array([1.0, 1.0, 1.0])
    objective_vector = np.array([1.0, 1.0, 1.0])
    s_const.evaluate(decision_vector, objective_vector)


def test_constraint_factory():
    pass
