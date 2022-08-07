import numpy as np
import pandas as pd
import pytest
from desdeo_problem.problem.Objective import (
    ObjectiveError,
    ObjectiveEvaluationResults,
    VectorDataObjective,
    VectorObjective,
)

# ============= utils ============


# just testing the basic functionality. Just return the vector as eval.
def evaluator(x):
    return x


# this should fail since we do not any evaluator to vectorDataObjective.
def evaluate_vec_data_obj():
    data = [["f1", 4.0], ["f2", 7.0]]
    df = pd.DataFrame(data, columns=["f1", "f2"])
    vec_data_obj = VectorDataObjective(["f1", "f2"], data=df)
    vec_data_obj.evaluate(np.array([1.1, 1.1, 1.1]))


# ============= TESTs ==========


def test_evalutation_fails():
    with pytest.raises(ObjectiveError):
        evaluate_vec_data_obj()


def test_obj():
    vec_obj = VectorObjective(["f1, f2, f3"], evaluator=evaluator)
    res = vec_obj.evaluate(np.array([1.1, 1.1, 1.1]))
    assert type(res) is ObjectiveEvaluationResults, "something went wrong"
