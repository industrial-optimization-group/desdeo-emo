"""This module implements methods for defining functions to scalarize vector valued functions.
These are knows as `Scalarizer`s. It also provides achievement scalarizing functions to be used
with the scalarizers.
"""

__all__ = [
    "AugmentedGuessASF",
    "MaxOfTwoASF",
    "PointMethodASF",
    "ReferencePointASF",
    "SimpleASF",
    "StomASF",
    "GuessASF",
    "DiscreteScalarizer",
    "Scalarizer",
    "Tchebycheff",
    "WeightedSum",
    "PBI",
    "reference_point_method_GLIDE",
    "AUG_GUESS_GLIDE",
    "GUESS_GLIDE",
    "AUG_STOM_GLIDE",
    "STOM_GLIDE",
    "NIMBUS_GLIDE",
    "PROJECT_GLIDE",
    "STEP_GLIDE",
    "Tchebycheff_GLIDE",
]

from desdeo_tools.scalarization.ASF import (
    AugmentedGuessASF,
    MaxOfTwoASF,
    PointMethodASF,
    ReferencePointASF,
    SimpleASF,
    StomASF,
    GuessASF,
)

from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer, Scalarizer
from desdeo_tools.scalarization.MOEADSF import Tchebycheff, WeightedSum, PBI

from desdeo_tools.scalarization.GLIDE_II import (
    reference_point_method_GLIDE,
    AUG_GUESS_GLIDE,
    GUESS_GLIDE,
    AUG_STOM_GLIDE,
    STOM_GLIDE,
    NIMBUS_GLIDE,
    PROJECT_GLIDE,
    STEP_GLIDE,
    Tchebycheff_GLIDE,
)
