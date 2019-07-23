import numpy as np
import pandas as pd
from pyDOE import lhs
from pyrvea.Problem.baseproblem import BaseProblem


class OptTestFunctions(BaseProblem):

    """Test functions for single/multi-objective problems to test
    the performance of evolutionary algorithms.

    See: https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters
    ----------
    name : str
        name of the test function
    num_of_variables : int
        number of decision variables
    num_of_objectives : int
        number of objectives
    num_of_constraints : int
        number of constraints
    upper_limits : float
        upper boundaries for test data
    lower_limits : float
        lower boundaries for test data
    """

    def __init__(
        self,
        name=None,
        num_of_variables=2,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):

        super(OptTestFunctions, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )

        # Define search domain for test functions
        self.test_f_limits = {
            "Sphere": (-5, 5),
            "Matyas": (-10, 10),
            "Himmelblau": (-5, 5),
            "Rastigrin": (-5.12, 5.12),
            "Three-hump camel": (-5, 5),
            "Goldstein-Price": (-2, 2),
            "LeviN13": (-10, 10),
            "SchafferN2": (-100, 100),
            "Coello_ex1": (0, 1),
            "Fonseca-Fleming": (-4, 4),
            "Kursawe": (-5, 5),
            "SchafferN1": (-100, 100),
        }

        if self.name in self.test_f_limits.keys():
            self.lower_limits = self.test_f_limits[self.name][0]
            self.upper_limits = self.test_f_limits[self.name][1]

    def __call__(self, x):
        return self.objectives(x)

    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.

        Parameters
        ----------
        decision_variables : np.ndarray
            The decision variables.

        Returns
        -------
        The objective functions.

        """

        self.num_of_variables = decision_variables.shape[0]

        # Single objective test functions
        self.num_of_objectives = 1

        if self.name == "Sphere":
            # Sphere function, -5 <= x <= 5

            x = np.asarray(decision_variables)
            self.obj_func = sum(x ** 2)

        elif self.name == "Matyas":

            # Matyas function, -10 <= x, y <= 10

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        elif self.name == "Himmelblau":

            # Himmelblau's function, -5 <= x, y <= 5

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

        elif self.name == "Rastigrin":
            # Rastigrin function, -5.12 <= x <= 5.12

            x = np.asarray(decision_variables)
            n = len(x)
            self.obj_func = 10 * n + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

        elif self.name == "Three-hump camel":
            # Three-hump camel function,  -5 <= x, y <= 5

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6) / 6 + x * y + y ** 2

        elif self.name == "Goldstein-Price":
            # Goldstein-Price function, -2 <= x, y <= 2

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = (
                1
                + (x + y + 1) ** 2
                * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * (y ** 2))
            ) * (
                30
                + (2 * x - 3 * y) ** 2
                * (18 - 32 * x + 12 * (x ** 2) + 48 * y - 36 * x * y + 27 * (y ** 2))
            )

        elif self.name == "LeviN13":
            # Levi function N.13, -10 <= x, y <= 10

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = (
                np.sin(3 * np.pi * x) ** 2
                + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
                + (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
            )

        elif self.name == "SchafferN2":
            # Schaffer function N. 2, -100 <= x, y <= 100

            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            self.obj_func = (
                0.5
                + (np.sin((x ** 2 - y ** 2) ** 2) - 0.5)
                / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
            )

        # Test functions for multi-objective optimization

        elif self.name == "Coello_ex1":
            self.num_of_objectives = 2
            x = np.asarray(decision_variables[0])
            y = np.asarray(decision_variables[1])
            a = 2
            q = 4
            f1 = x
            f2 = (1 + 10 * y) * (
                1
                - (x / (1 + 10 * y)) ** a
                - x / (1 + 10 * y) * np.sin(2 * np.pi * q * x)
            )

            self.obj_func = [f1, f2]

        elif self.name == "Kursawe":
            self.num_of_objectives = 2
            x1 = np.asarray(decision_variables[0])
            x2 = np.asarray(decision_variables[1])
            x3 = np.asarray(decision_variables[1])
            f1 = -10 * np.exp(-0.2 * np.sqrt(x1 ** 2 + x2 ** 2)) - 10 * np.exp(
                -0.2 * np.sqrt(x2 ** 2 + x3 ** 2)
            )
            f2 = (
                abs(x1) ** 0.8
                + 5.0 * np.sin(x1 ** 3)
                + abs(x2) ** 0.8
                + 5.0 * np.sin(x2 ** 3)
                + abs(x3) ** 0.8
                + 5.0 * np.sin(x3 ** 3)
            )
            self.obj_func = [f1, f2]

        elif self.name == "Fonseca-Fleming":
            self.num_of_objectives = 2
            f1 = 0
            f2 = 0

            for n in range(len(decision_variables)):
                f1 += (decision_variables[n] - 1 / np.sqrt(len(decision_variables))) ** 2
                f2 += (decision_variables[n] + 1 / np.sqrt(len(decision_variables))) ** 2

            f1 = 1 - np.exp(-f1)
            f2 = 1 - np.exp(-f2)

            self.obj_func = [f1, f2]

        elif self.name == "SchafferN1":
            self.num_of_objectives = 2
            x = np.asarray(decision_variables[0])

            f1 = x ** 2
            f2 = (x - 2) ** 2

            self.obj_func = [f1, f2]

        return self.obj_func

    def create_training_data(self, samples=150, method="random", seed=None):
        """Create training data for test functions.

        Parameters
        ----------
        samples : int
            number of samples
        method : str
            method to use in data creation. Possible values random, lhs, linear, linear+zeros, linear+reverse.
        seed : int
            if a number is given, random data will be seeded
        """

        np.random.seed(seed)
        training_data_input = None

        if method == "random":

            training_data_input = np.random.uniform(
                self.lower_limits, self.upper_limits, (samples, self.num_of_variables)
            )

        elif method == "lhs":

            training_data_input = lhs(self.num_of_variables, samples) * (
                abs(self.upper_limits) + abs(self.lower_limits)
            ) - abs(self.upper_limits)

        elif method == "linear":

            training_data_input = np.linspace(
                (self.lower_limits, self.lower_limits),
                (self.upper_limits, self.upper_limits),
                samples,
            )

        elif method == "linear+zeros":

            training_data_input = np.linspace(
                (self.lower_limits, self.lower_limits),
                (self.upper_limits, self.upper_limits),
                samples,
            )
            tmp = np.linspace(self.upper_limits, 0, samples)
            x2 = np.zeros_like(tmp)
            training_data_input = np.vstack(
                (np.hstack((tmp, tmp)), np.hstack((tmp, x2)))
            ).T

        elif method == "linear+reverse":

            np.linspace(
                (self.lower_limits, self.lower_limits),
                (self.upper_limits, self.upper_limits),
                samples,
            )
            tmp = np.linspace(self.upper_limits, 0, samples)
            x2 = np.flip(tmp)
            training_data_input = np.vstack(
                (np.hstack((tmp, tmp)), np.hstack((tmp, x2)))
            ).T

        training_data_output = np.asarray(
            [self.objectives(x) for x in training_data_input]
        )

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
