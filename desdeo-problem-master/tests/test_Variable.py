import numpy as np
import pytest
from desdeo_problem.problem.Variable import (
    Variable,
    VariableBuilderError,
    VariableError,
)

# ============ utils ===========

# this will fail, lower bound > upper bound
def make_bad_var():
    var = Variable("var_fail", 1.0, 2.0, -1.0)
    assert type(var) is Variable


# =============== TESTs ===========


def test_building_fails():
    with pytest.raises(VariableError):
        make_bad_var()


def test_basic_vars():
    test_var = Variable("var_1", 0.1)
    test_var_2 = Variable("var_2", 10.4, -1.0, 20.3)
    # test get_bounds
    assert test_var_2.get_bounds() == tuple((-1.0, 20.3))
    assert type(test_var) is Variable, "something went wrong"
    assert type(test_var_2) is Variable, "something went wrong"
