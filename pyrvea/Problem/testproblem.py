from optproblems import dtlz, zdt
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.Problem.test_functions import OptTestFunctions
import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.preprocessing import minmax_scale


class TestProblem(BaseProblem):
    """Test functions for single/multi-objective problems to test
    the performance of evolutionary algorithms.

    See: https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters
    ----------
    name : str
        Name of the test function.
    num_of_variables : int
        Number of decision variables.
    num_of_objectives : int
        Number of objectives.
    num_of_constraints : int
        Number of constraints.
    upper_limits : float
        Upper boundaries for test data.
    lower_limits : float
        Lower boundaries for test data.
    """

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1.0,
        lower_limits=0.0,
    ):

        super(TestProblem, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )
        if name == "ZDT1":
            self.obj_func = zdt.ZDT1()

        elif name == "ZDT2":
            self.obj_func = zdt.ZDT2()

        elif name == "ZDT3":
            self.obj_func = zdt.ZDT3()

        elif name == "ZDT4":
            self.obj_func = zdt.ZDT4()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT5":
            self.obj_func = zdt.ZDT5()

        elif name == "ZDT6":
            self.obj_func = zdt.ZDT6()

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
        else:
            self.obj_func = OptTestFunctions(name=self.name)
            self.num_of_variables = self.obj_func.num_of_variables
            self.num_of_objectives = self.obj_func.num_of_objectives
            self.lower_limits = self.obj_func.lower_limits
            self.upper_limits = self.obj_func.upper_limits

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

    def create_training_data(self, samples=150, method="random", seed=None):
        """Create training data for test functions.

        Parameters
        ----------
        samples : int
            Number of samples.
        method : str
            Method to use in data creation. Possible values random, lhs, linear.
        seed : int
            If a number is given, random data will be seeded.

        Returns
        -------
        dataset : Training data in a Pandas DataFrame, including all variables and objectives.
        x : List of variables
        y : List of objectives

        """

        np.random.seed(seed)
        training_data_input = None

        if method == "random":

            training_data_input = np.random.uniform(
                self.lower_limits, self.upper_limits, (samples, self.num_of_variables)
            )

        elif method == "lhs":

            training_data_input = lhs(self.num_of_variables, samples)
            training_data_input = np.round(
                minmax_scale(
                    training_data_input, (self.lower_limits, self.upper_limits)
                ),
                decimals=5,
            )

        elif method == "linear":

            training_data_input = np.linspace(
                self.lower_limits, self.upper_limits, samples
            )

        training_data_output = np.asarray(
            [self.objectives(x) for x in training_data_input]
        )
        if self.num_of_objectives == 1:
            training_data_output = training_data_output[:, None]

        # Convert numpy array into pandas dataframe, and make columns for it
        data = np.hstack((training_data_input, training_data_output))
        dataset = pd.DataFrame.from_records(data)
        x = []
        y = []
        for var in range(training_data_input.shape[1]):
            x.append("x" + str(var + 1))
        for obj in range(training_data_output.shape[1]):
            y.append("f" + str(obj + 1))
        dataset.columns = x + y

        np.random.seed(None)

        return dataset, x, y
