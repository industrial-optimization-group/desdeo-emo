class BaseProblem:
    """Base class for the problems."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """
        Pydocstring is ruthless.

        Parameters
        ----------
        name

        num_of_variables

        num_of_objectives

        num_of_constraints

        upper_limits

        lower_limits

        """
        self.name = name
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.num_of_constraints = num_of_constraints
        self.obj_func = []
        self.upper_limits = upper_limits
        self.lower_limits = lower_limits
        self.minimize = None

    def objectives(self, decision_variables):
        """Accept a sample. Return Objective values.

        Parameters
        ----------
        decision_variables
        """
        pass

    def constraints(self, decision_variables, objective_variables):
        """Accept a sample and/or corresponding objective values.

        Parameters
        ----------
        decision_variables
        objective_variables
        """
        pass

    def update(self):
        """Update the problem based on new information."""
        pass
