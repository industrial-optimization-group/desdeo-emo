import pandas as pd

from typing import Callable, List, Union


from desdeo_tools.interaction.validators import (
    validate_ref_point_with_ideal_and_nadir,
    validate_specified_solutions,
    validate_bounds,
)
from desdeo_tools.utilities.frozen import FrozenClass


class RequestError(Exception):
    """Raised when an error related to the Request class is encountered.
    """


class BaseRequest(FrozenClass):
    """The base class for all Request classes. Request classes are to be used
    to handle interaction between the user and the methods, as well as within
    various methods. This class is frozen, so no variables other than that
    already defined in current __init__ can be defined in derived classes.
    """

    def __init__(
        self,
        request_type: str,
        interaction_priority: str,
        content=None,
        request_id: int = None,
    ):
        """Initialize a BaseRequest class. This method contains a lot of
        boilerplate.

        Args:
            request_type (str): The type of request. Currently, one of ["print",
                "simple_plot", "reference_point_preference",
                "classification_preference"].
            interaction_priority (str): The priority of preference, as decided
                by the method. One of ["no_interaction", "not_required",
                "recommended", "required"], with trivial meanings.
            content ([type], optional): The data relevant to the request packet.
                For example, if the request type is print, content may contain
                strings to be printed. Typically a dict. Defaults to None.
            request_id (int, optional): A unique identifier. Defaults to None.

        Raises:
            RequestError: If request type is not recognized
            RequestError: If request priority is not recognized
            RequestError: If request id is not an integer
        """
        acceptable_types = [
            "print",
            "simple_plot",
            "reference_point_preference",
            "classification_preference",
            "preferred_solution_preference",
            "non_preferred_solution_preference",
            "bound_preference",
        ]
        priority_types = ["no_interaction", "not_required", "recommended", "required"]
        if request_type not in acceptable_types:
            msg = f"Request type should be one of {acceptable_types}"
            raise RequestError(msg)
        if interaction_priority not in priority_types:
            msg = f"Request priority should be one of {priority_types}"
            raise RequestError(msg)
        if not isinstance(request_id, (int, type(None))):
            msg = "Request id should be int or None"
            raise RequestError(msg)
        # Attributes
        self._request_type: str = request_type
        self._interaction_priority: str = interaction_priority  # Example: one of:
        self._request_id: int = request_id  # Some random number as id
        self._content = content
        self._response = None
        #  Freezing this class
        self._freeze()

    @property
    def request_type(self):
        return self._request_type

    @property
    def interaction_priority(self):
        return self._interaction_priority

    @property
    def request_id(self):
        return self._request_id

    @property
    def content(self):
        return self._content

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self):
        # Code to validate the response
        return


class PrintRequest(BaseRequest):
    """Methods can use this request class to send out textual information to be
    displayed to the decision maker. This could be a single message in the form
    of a string, or multiple messages in a list of strings. The method of
    displaying these messages is left to the UI.
    """

    def __init__(self, message: Union[str, List[str]], request_id: int = None):
        """Initialise the PrintRequest.

        Args:
            message (Union[str, List[str]]): A single message (str) or a list of
                messages to be displayed to the decision maker
            request_id (int, optional): A unique identifier for this request.
                Defaults to None.

        Raises:
            RequestError: If message is not a str or a list
            RequestError: If message is a list but one or more elements are not
                str.
        """
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        super().__init__(
            request_type="print",
            interaction_priority="no_interaction",
            content=message,
            request_id=request_id,
        )


class SimplePlotRequest(BaseRequest):
    """Methods can use this request class to send out some data to be shown to
    the decision maker (typically in the form of a plot). This data is usually a
    set of solutions, stored in the content variable of this class. The manner
    of visualization is left to the UI.

    content is a dict that contains the following keys:
        "data" (pandas.DataFrame): The data to be plotted.
        "dimensional_data" (pandas.Dataframe): The data contained in this key can be
            used to scale the data to be plotted.
        "chart_title" (str): A recommended title for the visualization.
        "message" (Union[str, List[str]]): A message or list of messages to be
            displayed to the decision maker.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        message: Union[str, List[str]],
        dimensions_data: pd.DataFrame = None,
        chart_title: str = None,
        request_id=None,
    ):
        """Initialize the request packet

        Args:
            data (pd.DataFrame): The data to be plotted.
            message (Union[str, List[str]]): A message or list of messages to be
                displayed to the decision maker.
            dimensions_data (pd.DataFrame, optional): Data used to used to scale
                the data to be plotted. Defaults to None.
            chart_title (str, optional): A recommended title for the
                visualization. Defaults to None.
            request_id ([type], optional): A unique identifier. Defaults to None.

        Raises:
            RequestError: data is not a pandas DataFrame.
            RequestError: dimensions_data is not a pandas DataFrame or None.
            RequestError: A mismatch in the column names of data and
                dimensions_data.
            RequestError: If dimensions_data DataFrame contains indices other
                that "minimize", "ideal", or "nadir".
            RequestError: If chart_title is not str or None.
            RequestError: If message is not a str or a list.
            RequestError: If message is a list but one or more elements are not
                str.
        """
        acceptable_dimensions_data_indices = [
            "minimize",  # 1 if minimized, -1 if maximized
            "ideal",
            "nadir",
        ]
        if not isinstance(data, pd.DataFrame):
            msg = (
                f"Provided data to be plotted should be in a pandas dataframe, with"
                f"columns names being the same as objective names.\n"
                f"Provided data is of type: {type(data)}"
            )
            raise RequestError(msg)
        if not isinstance(dimensions_data, (pd.DataFrame, type(None))):
            msg = (
                f"Dimensional data should be in a pandas dataframe.\n"
                f"Provided data is of type: {type(dimensions_data)}"
            )
            raise RequestError(msg)
        if not all(data.columns == dimensions_data.columns):
            msg = (
                f"Mismatch in column names of data and dimensions_data.\n"
                f"Column names in data: {data.columns}"
                f"Column names in dimensions_data: {dimensions_data.columns}"
            )
            raise RequestError(msg)
        rouge_indices = [
            index
            for index in dimensions_data.index
            if index not in acceptable_dimensions_data_indices
        ]
        if rouge_indices:
            msg = (
                f"dimensions_data should only contain the following indices:\n"
                f"{acceptable_dimensions_data_indices}\n"
                f"The dataframe provided contains the following unsupported indices:\n"
                f"{rouge_indices}"
            )
            raise RequestError(msg)
        if not isinstance(chart_title, (str, type(None))):
            msg = (
                f"Chart title should be a string. Provided chart type is:"
                f"{type(chart_title)}"
            )
            raise RequestError(msg)
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "data": data,
            "dimensions_data": dimensions_data,
            "chart_title": chart_title,
            "message": message,
        }
        super().__init__(
            request_type="simple_plot",
            interaction_priority="no_interaction",
            content=content,
            request_id=request_id,
        )


class ReferencePointPreference(BaseRequest):
    """Methods can use this request class to ask the DM to provide their preferences
    in the form of a reference point. This reference point is validated according to
    the needs of the method that initializes this class object, before the reference
    point can be accepted in the response variable.
    """

    def __init__(
        self,
        dimensions_data: pd.DataFrame,
        message: str = None,
        interaction_priority: str = "required",
        preference_validator: Callable = None,
        request_id: int = None,
    ):
        """Initialize the request class.

        Args:
            dimensions_data (pd.DataFrame): Minimal data that should be shown to
                the decision maker. If a lot of data needs to be shown (i.e., with a
                visualization), use SimplePlotRequest or related classes for that
                purpose, and this class for the interaction with the decision maker.
            message (str, optional): Message to be displayed to a decision
                maker. Defaults to None.
            interaction_priority (str, optional): The importance of the
                interaction as decided by the method. If equal to "required", the
                method will not continue without a DM preference. If equal to
                "recommended", the interaction is recommended, but not required for
                the continuation of the method. The case "not_required" is similar
                to "recommended". Defaults to "required".
            preference_validator (Callable, optional): A callable function that
                tests whether a reference point provided by the DM is valid or not.
                Defaults to None.
            request_id (int, optional): A unique identifier. Defaults to None.

        Raises:
            RequestError: dimensions_data is not a pandas DataFrame.
            RequestError: If dimensions_data DataFrame contains indices other
                that "minimize", "ideal", or "nadir".
            RequestError: If message is not a str or a list.
            RequestError: If message is a list but one or more elements are not
                str.
        """
        if message is None:
            message = (
                f"Please provide a reference point better than:\n"
                f"{dimensions_data.loc['nadir'].values.tolist()},\n"
                f"but worse than:\n"
                f"{dimensions_data.loc['ideal'].values.tolist()}"
            )
        if preference_validator is None:
            preference_validator = validate_ref_point_with_ideal_and_nadir
        acceptable_dimensions_data_indices = [
            "minimize",  # 1 if minimized, -1 if maximized
            "ideal",
            "nadir",
        ]
        if not isinstance(dimensions_data, (pd.DataFrame, type(None))):
            msg = (
                f"Dimensional data should be in a pandas dataframe.\n"
                f"Provided data is of type: {type(dimensions_data)}"
            )
            raise RequestError(msg)
        rouge_indices = [
            index
            for index in dimensions_data.index
            if index not in acceptable_dimensions_data_indices
        ]
        if rouge_indices:
            msg = (
                f"dimensions_data should only contain the following indices:\n"
                f"{acceptable_dimensions_data_indices}\n"
                f"The dataframe provided contains the following unsupported indices:\n"
                f"{rouge_indices}"
            )
            raise RequestError(msg)
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "dimensions_data": dimensions_data,
            "message": message,
            "validator": preference_validator,
        }
        super().__init__(
            request_type="reference_point_preference",
            interaction_priority=interaction_priority,
            content=content,
            request_id=request_id,
        )

    @BaseRequest.response.setter
    def response(self, value: pd.DataFrame):
        """Validate user preference. Accept if it is valid.

        Args:
            value (pd.DataFrame): The user preference in the form of a pandas DataFrame

        Raises:
            RequestError: If reference point is not provided in a pandas DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            msg = "Reference should be provided in a pandas dataframe format"
            raise RequestError(msg)
        self.content["validator"](
            reference_point=value, dimensions_data=self.content["dimensions_data"]
        )
        self._response = value


class PreferredSolutionPreference(BaseRequest):
    """Methods can use this class to ask the Decision maker to provide their preferences in form of preferred solution(s).
    """

    def __init__(
        self,
        n_solutions: int,
        message: str = None,
        interaction_priority: str = "required",
        preference_validator: Callable = None,
        request_id: int = None,
    ):
        """Initialize preference-class with information about problem.

        Args:
            n_solutions (int): Number of solutions in total.
            message (str): Message to be displayed to the Decision maker.
            interaction_priority (str): Level of priority.
            preference_validator (Callable): Function that validates the Decision maker's preferences.
            request_id (int): Identification number of request.
        """

        self._n_solutions = n_solutions

        if message is None:
            message = (
                "Please specify preferred solution(s) by their index as 'preferred_solutions_indices', so that the "
                "indices start at 0. Please specify the index/indices in a list."
            )

        if preference_validator is None:
            preference_validator = validate_specified_solutions

        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {"message": message, "validator": preference_validator}
        super().__init__(
            request_type="preferred_solution_preference",
            interaction_priority=interaction_priority,
            content=content,
            request_id=request_id,
        )

    @BaseRequest.response.setter
    def response(self, value):
        # validate the response
        self.content["validator"](indices=value, n_solutions=self._n_solutions)
        self._response = value


class NonPreferredSolutionPreference(BaseRequest):
    """Methods can use this class to ask the Decision maker to provide their preferences in form of non-preferred
    solution(s).
    """

    def __init__(
        self,
        n_solutions: int,
        message: str = None,
        interaction_priority: str = "required",
        preference_validator: Callable = None,
        request_id: int = None,
    ):
        """Initialize preference-class with information about problem.

        Args:
            n_solutions (int): Number of solutions in total.
            message (str): Message to be displayed to the Decision maker.
            interaction_priority (str): Level of priority.
            preference_validator (Callable): Function that validates the Decision maker's preferences.
            request_id (int): Identification number of request.
        """

        self._n_solutions = n_solutions

        if message is None:
            message = (
                "Please specify non-preferred solution(s) by their index as 'non-preferred_solutions_indices', so that "
                "the indices start at 0. Please specify the index/indices in a list."
            )

        if preference_validator is None:
            preference_validator = validate_specified_solutions

        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {"message": message, "validator": preference_validator}
        super().__init__(
            request_type="non_preferred_solution_preference",
            interaction_priority=interaction_priority,
            content=content,
            request_id=request_id,
        )

    @BaseRequest.response.setter
    def response(self, value):
        # validate the response
        self.content["validator"](indices=value, n_solutions=self._n_solutions)
        self._response = value


class BoundPreference(BaseRequest):
    """Methods can use this class to ask the Decision maker to provide their preferences in form of preferred lower and
    upper bounds for objective values.
    """

    def __init__(
        self,
        dimensions_data: pd.DataFrame,
        n_objectives: int,
        message: str = None,
        interaction_priority: str = "required",
        preference_validator: Callable = None,
        request_id: int = None,
    ):
        """Initialize preference-class with information about problem.

        Args:
            dimensions_data (pd.DataFrame): DataFrame including information whether an objective is minimized or
                maximized, for each objective. In addition, includes ideal and nadir vectors.
            n_objectives (int): Number of objectives in problem.
            message (str): Message to be displayed to the Decision maker.
            interaction_priority (str): Level of priority.
            preference_validator (Callable): Function that validates the Decision maker's preferences.
            request_id (int): Identification number of request.
        """

        self._n_objectives = n_objectives

        if message is None:
            message = (
                "Please specify desired lower and upper bound for each objective as 'bounds', starting from the first "
                "objective and ending with the last one. Please specify the bounds as a numpy array containing lists, "
                "so that the first item of list is the lower bound and the second the upper bound, for each objective."
                "For example: numpy.array([[1, 2], [2, 5], [0, 3.5]]), for problem with three objectives."
                "Ideal vector: {}\nNadir vector: {}.".format(
                    dimensions_data.loc["ideal"].values.tolist(),
                    dimensions_data.loc["nadir"].values.tolist(),
                )
            )

        if preference_validator is None:
            preference_validator = validate_bounds

        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    "Message/s to be printed should be string or list of strings"
                    "Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "dimensions_data": dimensions_data,
            "message": message,
            "validator": preference_validator,
        }
        super().__init__(
            request_type="bound_preference",
            interaction_priority=interaction_priority,
            content=content,
            request_id=request_id,
        )

    @BaseRequest.response.setter
    def response(self, value):
        # validate the response
        self.content["validator"](
            dimensions_data=self.content["dimensions_data"],
            bounds=value,
            n_objectives=self._n_objectives,
        )
        self._response = value
