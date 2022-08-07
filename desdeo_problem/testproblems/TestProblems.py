from optproblems import zdt, dtlz
from desdeo_problem.problem.Variable import variable_builder
from desdeo_problem.problem.Objective import VectorObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Problem import ProblemError


def test_problem_builder(name: str, n_of_variables: int = None, n_of_objectives: int = None) -> MOProblem:
    """Build test problems. Currently supported: ZDT1-4, ZDT6, and DTLZ1-7.

    Args:
        name (str): Name of the problem in all caps. For example: "ZDT1", "DTLZ4", etc.
        n_of_variables (int, optional): Number of variables. Required for DTLZ problems,
            but can be skipped for ZDT problems as they only support one variable value.
        n_of_objectives (int, optional): Required for DTLZ problems,
            but can be skipped for ZDT problems as they only support one variable value.

    Raises:
        ProblemError: When one of many issues occur while building the MOProblem
            instance.

    Returns:
        MOProblem: The test problem object
    """
    problems = {
        "ZDT1": zdt.ZDT1,
        "ZDT2": zdt.ZDT2,
        "ZDT3": zdt.ZDT3,
        "ZDT4": zdt.ZDT4,
        "ZDT5": zdt.ZDT5,
        "ZDT6": zdt.ZDT6,
        "DTLZ1": dtlz.DTLZ1,
        "DTLZ2": dtlz.DTLZ2,
        "DTLZ3": dtlz.DTLZ3,
        "DTLZ4": dtlz.DTLZ4,
        "DTLZ5": dtlz.DTLZ5,
        "DTLZ6": dtlz.DTLZ6,
        "DTLZ7": dtlz.DTLZ7,
    }
    num_var = {"ZDT1": 30, "ZDT2": 30, "ZDT3": 30, "ZDT4": 10, "ZDT6": 10}
    if not (name in problems.keys()):
        msg = "Specified Problem not yet supported.\n The supported problems are:" + str(problems.keys())
        raise ProblemError(msg)
    if "ZDT" in name:
        if n_of_variables is None:
            n_of_variables = num_var[name]
        if n_of_objectives is None:
            n_of_objectives = 2
        if not (n_of_variables == num_var[name]):
            msg = (
                name
                + " problem has been limited to "
                + str(num_var[name])
                + " variables. Number of variables recieved = "
                + str(n_of_variables)
            )
            raise ProblemError(msg)
        if not (n_of_objectives == 2):
            msg = (
                "ZDT problems can only have 2 objectives. " + "Number of objectives recieved = " + str(n_of_objectives)
            )
            raise ProblemError(msg)
        obj_func = problems[name]()
    elif "DTLZ" in name:
        if (n_of_variables is None) or (n_of_objectives is None):
            msg = "Please provide both number of variables and objectives" + " for the DTLZ problems"
            raise ProblemError(msg)
        obj_func = problems[name](n_of_objectives, n_of_variables)
    else:
        msg = "How did you end up here?"
        raise ProblemError(msg)
    lower_limits = obj_func.min_bounds
    upper_limits = obj_func.max_bounds
    var_names = ["x" + str(i + 1) for i in range(n_of_variables)]
    obj_names = ["f" + str(i + 1) for i in range(n_of_objectives)]
    variables = variable_builder(
        names=var_names,
        initial_values=lower_limits,
        lower_bounds=lower_limits,
        upper_bounds=upper_limits,
    )

    # Because optproblems can only handle one objective at a time
    def modified_obj_func(x):
        if isinstance(x, list):
            if len(x) == n_of_variables:
                return [obj_func(x)]
            elif len(x[0]) == n_of_variables:
                return list(map(obj_func, x))
        else:
            if x.ndim == 1:
                return [obj_func(x)]
            elif x.ndim == 2:
                return list(map(obj_func, x))
        raise TypeError("Unforseen problem, contact developer")

    objective = VectorObjective(name=obj_names, evaluator=modified_obj_func)
    problem = MOProblem([objective], variables, None)
    return problem
