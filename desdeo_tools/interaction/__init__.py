"""This module contains classes implementing different interactions to be
used to communicate between different optimization algorithms and users.
"""

__all__ = [
    "PrintRequest",
    "SimplePlotRequest",
    "ReferencePointPreference",
    "PreferredSolutionPreference",
    "NonPreferredSolutionPreference",
    "BoundPreference",
    "validate_ref_point_data_type",
    "validate_ref_point_dimensions",
    "validate_ref_point_with_ideal",
    "validate_ref_point_with_ideal_and_nadir",
    "validate_with_ref_point_nadir",
    "validate_specified_solutions",
    "validate_bounds",
]

from desdeo_tools.interaction.request import (
    PrintRequest,
    ReferencePointPreference,
    SimplePlotRequest,
    PreferredSolutionPreference,
    NonPreferredSolutionPreference,
    BoundPreference,
)

from desdeo_tools.interaction.validators import (
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
    validate_ref_point_with_ideal_and_nadir,
    validate_with_ref_point_nadir,
    validate_specified_solutions,
    validate_bounds,
)
