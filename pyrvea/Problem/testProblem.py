from optproblems import dtlz, zdt
from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.Problem.test_functions import OptTestFunctions


class testProblem(baseProblem):
    """Defines the problem."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """Pydocstring is ruthless.

        Args:
            name:
            num_of_variables:
            num_of_objectives:
            num_of_constraints:
            upper_limits:
            lower_limits:
        """
        super(testProblem, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )
        if name == "ZDT1":
            self.obj_func = zdt.ZDT1()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT2":
            self.obj_func = zdt.ZDT2()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT3":
            self.obj_func = zdt.ZDT3()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT4":
            self.obj_func = zdt.ZDT4()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT5":
            self.obj_func = zdt.ZDT5()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT6":
            self.obj_func = zdt.ZDT6()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ1":
            self.obj_func = dtlz.DTLZ1(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ2":
            self.obj_func = dtlz.DTLZ2(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ3":
            self.obj_func = dtlz.DTLZ3(num_of_objectives, num_of_variables)
            self.lower_limits = 0
            self.upper_limits = 1
        elif name == "DTLZ4":
            self.obj_func = dtlz.DTLZ4(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ5":
            self.obj_func = dtlz.DTLZ5(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ6":
            self.obj_func = dtlz.DTLZ6(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ7":
            self.obj_func = dtlz.DTLZ7(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds

    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.

        Args:
            decision_variables:
        """
        return self.obj_func(decision_variables)

    def constraints(self, decision_variables, objective_variables):
        """Calculate constraint violation.

        Args:
            decision_variables:
            objective_variables:
        """
        print("Error: Constraints not supported yet.")
