import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Raised when an error related to the validation is encountered.
    """


def validate_ref_point_with_ideal_and_nadir(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    validate_ref_point_data_type(reference_point)
    validate_ref_point_with_ideal(dimensions_data, reference_point)
    validate_with_ref_point_nadir(dimensions_data, reference_point)


def validate_ref_point_with_ideal(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    ideal_fitness = dimensions_data.loc["ideal"] * dimensions_data.loc["minimize"]
    ref_point_fitness = reference_point * dimensions_data.loc["minimize"]
    if not (ideal_fitness <= ref_point_fitness).all(axis=None):
        problematic_columns = ideal_fitness.index[
            (ideal_fitness > ref_point_fitness).values.tolist()[0]
        ].values
        msg = (
            f"Reference point should be worse than or equal to the ideal point\n"
            f"The following columns have problematic values: {problematic_columns}"
        )
        raise ValidationError(msg)


def validate_with_ref_point_nadir(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    nadir_fitness = dimensions_data.loc["nadir"] * dimensions_data.loc["minimize"]
    ref_point_fitness = reference_point * dimensions_data.loc["minimize"]
    if not (ref_point_fitness <= nadir_fitness).all(axis=None):
        problematic_columns = nadir_fitness.index[
            (nadir_fitness < ref_point_fitness).values.tolist()[0]
        ].values
        msg = (
            f"Reference point should be better than or equal to the nadir point\n"
            f"The following columns have problematic values: {problematic_columns}"
        )
        raise ValidationError(msg)


def validate_ref_point_dimensions(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    if not dimensions_data.shape[1] == reference_point.shape[1]:
        msg = (
            f"There is a mismatch in the number of columns of the dataframes.\n"
            f"Columns in dimensions data: {dimensions_data.columns}\n"
            f"Columns in the reference point provided: {reference_point.columns}"
        )
        raise ValidationError(msg)
    if not all(dimensions_data.columns == reference_point.columns):
        msg = (
            f"There is a mismatch in the column names of the dataframes.\n"
            f"Columns in dimensions data: {dimensions_data.columns}\n"
            f"Columns in the reference point provided: {reference_point.columns}"
        )
        raise ValidationError(msg)


def validate_ref_point_data_type(reference_point: pd.DataFrame):
    for dtype in reference_point.dtypes:
        if not pd.api.types.is_numeric_dtype(dtype):
            msg = (
                f"Type of data in reference point dataframe should be numeric.\n"
                f"Provided datatype: {dtype}"
            )
            raise ValidationError(msg)


def validate_specified_solutions(indices: np.ndarray, n_solutions: int) -> None:
    """Validate the Decision maker's choice of preferred/non-preferred solutions.

    Args:
        indices (np.ndarray): Index/indices of preferred solutions specified by the Decision maker.
        n_solutions (int): Number of solutions in total.

    Returns:

    Raises:
        ValidationError: In case the preference is invalid.
    """

    if indices.shape[0] < 1:
        raise ValidationError("Please specify at least one (non-)preferred solution.")
    if not isinstance(indices, (np.ndarray, list)):
        raise ValidationError("Please specify index/indices of (non-)preferred solutions in a list, even if there is only "
                           "one.")
    if not all(0 <= i <= (n_solutions - 1) for i in indices):
        msg = "indices of (non-)preferred solutions should be between 0 and {}. Current indices are {}." \
            .format(n_solutions - 1, indices)
        raise ValidationError(msg)


def validate_bounds(dimensions_data: pd.DataFrame, bounds: np.ndarray, n_objectives: int) -> None:
    """Validate the Decision maker's desired lower and upper bounds for objective values.

    Args:
        dimensions_data (pd.DataFrame): DataFrame including information whether an objective is minimized or
            maximized, for each objective. In addition, includes ideal and nadir vectors.
        bounds (np.ndarray): Desired lower and upper bounds for each objective.
        n_objectives (int): Number of objectives in problem.

    Returns:

    Raises:
        ValidationError: In case desired bounds are invalid.
    """

    if not isinstance(bounds, np.ndarray):
        msg = "Please specify bounds as a numpy array. Current type: {}.".format(type(bounds))
        raise ValidationError(msg)
    if len(bounds) != n_objectives:
        msg = "Length of 'bounds' ({}) must be the same as number of objectives ({}).".format(len(bounds), n_objectives)
        raise ValidationError(msg)
    if not all(isinstance(b, (np.ndarray, list)) for b in bounds):
        print(type(bounds[0]))
        msg = "Please give bounds for each objective in a list."
        raise ValidationError(msg)
    if any(len(b) != 2 for b in bounds):
        msg = "Length of each item of 'bounds' must 2, containing the lower and upper bound for an objective."
        raise ValidationError(msg)
    if any(b[0] > b[1] for b in bounds):
        msg = "Lower bound cannot be greater than upper bound. Please specify lower bound first, then upper bound."
        raise ValidationError(msg)

    # check that bounds are within ideal and nadir points for each objective
    for i, b in enumerate(bounds):
        if dimensions_data.loc['minimize'].values.tolist()[i] == 1:  # minimized objectives
            if dimensions_data.loc['ideal'].values.tolist()[i] is not None:
                if b[0] < dimensions_data.loc['ideal'].values.tolist()[i]:
                    msg = "Lower bound cannot be lower than ideal value for objective. Ideal vector: {}." \
                        .format(dimensions_data.loc['ideal'].values.tolist())
                    raise ValidationError(msg)
            if dimensions_data.loc['nadir'].values.tolist()[i] is not None:
                if b[1] > dimensions_data.loc['nadir'].values.tolist()[i]:
                    msg = "Upper bound cannot be higher than nadir value for objective. Nadir vector: {}." \
                        .format(dimensions_data.loc['nadir'].values.tolist())
                    raise ValidationError(msg)

        else:  # maximized objectives:
            if dimensions_data.loc['ideal'].values.tolist()[i] is not None:
                if b[1] > dimensions_data.loc['ideal'].values.tolist()[i]:
                    msg = "Upper bound cannot be higher than ideal value for objective. Ideal vector: {}." \
                        .format(dimensions_data.loc['ideal'].values.tolist())
                    raise ValidationError(msg)
            if dimensions_data.loc['nadir'].values.tolist()[i] is not None:
                if b[0] < dimensions_data.loc['nadir'].values.tolist()[i]:
                    msg = "Lower bound cannot be lower than nadir value for objective. Nadir vector: {}." \
                        .format(dimensions_data.loc['nadir'].values.tolist())
                    raise ValidationError(msg)
