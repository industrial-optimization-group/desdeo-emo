import pytest
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

# smh wrong with importing test_problem_builder..

# fixtures are kind of parameters to give to the test functions like test_problem_builder


@pytest.fixture
def name():
    name = "ZDT1"
    return name


# problem_name = "DTLZ1"
# problem = test_problem_builder(problem_name, n_of_variables=7, n_of_objectives=3)
# problem_name = "DTLZ2"
# problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
# problem_name = "DTLZ4"
# problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
# problem_name = "DTLZ6"
# problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)
# problem_name = "DTLZ7"
# problem = test_problem_builder(problem_name, n_of_variables=22, n_of_objectives=3)

# tälle voi antaa tavaraa. "param nimet", aloitusarvot, lopetusarvot, esim. x*y == z
# @pytest.mark.parametrize("x,y,z", [(10,20,200), (20,40,200)])


@pytest.fixture
def dtlz_params():
    name = "DTLZ4"
    vars = 12
    objs = 3
    return [name, vars, objs]


def test_zdt(name):
    prob = test_problem_builder(name)
    assert type(prob) == MOProblem


def test_dtlz(dtlz_params):
    prob = test_problem_builder(dtlz_params[0], dtlz_params[1], dtlz_params[2])
    assert type(prob) == MOProblem
