import matplotlib.pyplot as plt
import numpy as np
import pytest
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator


# This does not run yet. Trying to call the plots without getting stuck on the plot outputs.
# monkeypatch is something you want to use
@pytest.fixture(scope="function")
def test_plots(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    problem.plot_problem_instance()
    po_set = problem.plot_pareto_set_members(100)
    assert len(po_set) > 0, "did not return po_set"
    problem.plot_landscape_for_single_objective(0, 100)
    # problem.plot_dominance_landscape(100) # not implemented yet


def test_type_0_DBMOPP():
    n_objectives = 3
    n_variables = 6
    n_local_pareto_regions = 0
    n_dominance_res_regions = 0
    n_global_pareto_regions = 3
    const_space = 0.0
    pareto_set_type = 0
    constraint_type = 0
    ndo = 0
    neutral_space = 0.0

    problem = DBMOPP_generator(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_dominance_res_regions,
        n_global_pareto_regions,
        const_space,
        pareto_set_type,
        constraint_type,
        ndo,
        False,
        False,
        neutral_space,
        10000,
    )
    x, point = problem.get_Pareto_set_member()
    assert (
        x.any() or point.any() is not None
    ), "x or point are None in get_Pareto_set_member"
    x = np.array(np.random.rand(5, n_variables))
    moproblem = problem.generate_problem()
    assert moproblem is not None, "moproblem was not formed"
    moproblem.evaluate(x)


def test_type_1_DBMOPP():
    n_objectives = 4
    n_variables = 8
    n_local_pareto_regions = 2
    n_dominance_res_regions = 2
    n_global_pareto_regions = 6
    const_space = 0.0
    pareto_set_type = 1
    constraint_type = 6
    ndo = 0
    neutral_space = 0.0

    problem = DBMOPP_generator(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_dominance_res_regions,
        n_global_pareto_regions,
        const_space,
        pareto_set_type,
        constraint_type,
        ndo,
        False,
        False,
        neutral_space,
        10000,
    )
    print("Initializing works!")

    x, point = problem.get_Pareto_set_member()

    assert (
        x.any() or point.any() is not None
    ), "x or point are None in get_Pareto_set_member"

    x = np.array(np.random.rand(5, n_variables))
    moproblem = problem.generate_problem()
    moproblem.evaluate(x)

    assert moproblem is not None, "moproblem was not formed"


def test_type_2_DBMOPP():
    n_objectives = 5
    n_variables = 4
    n_local_pareto_regions = 2
    n_dominance_res_regions = 1
    n_global_pareto_regions = 3
    const_space = 0.4
    pareto_set_type = 2
    constraint_type = 4
    ndo = 1
    neutral_space = 0.1

    problem = DBMOPP_generator(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_dominance_res_regions,
        n_global_pareto_regions,
        const_space,
        pareto_set_type,
        constraint_type,
        ndo,
        False,
        False,
        neutral_space,
        10000,
    )
    print("Initializing works!")

    x, point = problem.get_Pareto_set_member()

    assert (
        x.any() or point.any() is not None
    ), "x or point are None in get_Pareto_set_member"

    x = np.array(np.random.rand(5, n_variables))
    moproblem = problem.generate_problem()
    moproblem.evaluate(x)

    assert moproblem is not None, "moproblem was not formed"
    return problem


problem = test_type_2_DBMOPP()
