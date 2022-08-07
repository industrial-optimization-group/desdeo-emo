import pytest
import numpy as np

from desdeo_tools.solver.ScalarSolver import (
    ScalarMethod,
    ScalarMinimizer,
    DiscreteMinimizer,
    DiscreteScalarizer,
    ScalarSolverException,
)
from desdeo_tools.scalarization.ASF import PointMethodASF


def simple_problem(xs: np.ndarray):
    return xs[0] * 2 - xs[1] + xs[2]


def simple_constr(xs: np.ndarray):
    if xs[0] > 0.2:
        con_1 = 1
    else:
        con_1 = -1

    if xs[2] < 0.2:
        con_2 = 1
    else:
        con_2 = -1

    return np.array([con_1, con_2])


def dummy_minimizer(fun, x0, bounds, constraints=None):
    res_dict = {}

    if constraints is not None:
        con_vals = constraints(x0)
        if np.all(con_vals > 0):
            res_dict["success"] = True
        else:
            res_dict["success"] = False
    else:
        res_dict["success"] = True

    res_dict["x"] = x0

    res_dict["message"] = "I just retruned the initial guess as the optimal solution."

    return res_dict


def test_dummy_no_cons():
    method = ScalarMethod(dummy_minimizer)
    solver = ScalarMinimizer(
        simple_problem, np.array([[0, 0, 0], [1, 1, 1]]), None, method
    )

    x0 = np.array([0.5, 0.5, 0.5])
    res = solver.minimize(x0)

    assert np.array_equal(res["x"], x0)
    assert res["success"]
    assert (
        res["message"] == "I just retruned the initial guess as the optimal solution."
    )


def test_dummy_cons():
    method = ScalarMethod(dummy_minimizer)
    solver = ScalarMinimizer(
        simple_problem, np.array([[0, 0, 0], [1, 1, 1]]), simple_constr, method
    )

    res = solver.minimize(np.array([0.5, 0.5, 0.1]))

    assert res["success"]

    res = solver.minimize(np.array([0.5, 0.5, 0.5]))

    assert not res["success"]


def test_scipy_de_cons():
    solver = ScalarMinimizer(
        simple_problem,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T,
        simple_constr,
        "scipy_de",
    )

    res = solver.minimize(None)

    assert res["success"]

    assert np.all(np.array(res["constr"]) >= 0)


def test_scipy_minimize_cons():
    solver = ScalarMinimizer(
        simple_problem,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T,
        simple_constr,
        "scipy_minimize",
    )

    res = solver.minimize(np.array([0.21, 0.999, 0.001]))

    assert not res["success"]


def test_discrete_solver():
    ideal = np.array([0, 0, 0, 0])
    nadir = np.array([1, 1, 1, 1])

    asf = PointMethodASF(nadir, ideal)
    dscalarizer = DiscreteScalarizer(asf, {"reference_point": nadir})
    dminimizer = DiscreteMinimizer(dscalarizer)

    non_dominated_points = np.array(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.4, 0.2, 0.6, 0.8],
            [0.6, 0.4, 0.2, 0.8],
            [0.4, 0.8, 0.6, 0.2],
        ]
    )

    # first occurrence
    res_ind = dminimizer.minimize(non_dominated_points)["x"]
    assert res_ind == 0

    dscalarizer._scalarizer_args = {"reference_point": np.array([0.6, 0.4, 0.2, 0.8])}
    res_ind = dminimizer.minimize(non_dominated_points)["x"]

    assert res_ind == 2


def test_discrete_solver_with_con():
    ideal = np.array([0, 0, 0, 0])
    nadir = np.array([1, 1, 1, 1])

    asf = PointMethodASF(nadir, ideal)
    con = lambda x: x[:, 0] > 0.2
    dscalarizer = DiscreteScalarizer(asf, {"reference_point": nadir})
    dminimizer = DiscreteMinimizer(dscalarizer, constraint_evaluator=con)

    non_dominated_points = np.array(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.4, 0.2, 0.6, 0.8],
            [0.6, 0.4, 0.2, 0.8],
            [0.4, 0.8, 0.6, 0.2],
        ]
    )

    # first occurrence with first point invalid
    res_ind = dminimizer.minimize(non_dominated_points)["x"]

    assert res_ind == 1

    # first point as closest, but invalid
    dscalarizer._scalarizer_args = {"reference_point": np.array([0.2, 0.4, 0.6, 0.8])}
    res_ind = dminimizer.minimize(non_dominated_points)["x"]

    assert res_ind == 1

    # all points invalid
    dminimizer._constraint_evaluator = lambda x: x[:, 0] > 1.0

    with pytest.raises(ScalarSolverException):
        _ = dminimizer.minimize(non_dominated_points)


if __name__ == "__main__":
    solver = ScalarMinimizer(
        simple_problem,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T,
        simple_constr,
        "scipy_de",
    )

    res = solver.minimize(np.array([0.21, 0.999, 0.001]))

    print(res)
