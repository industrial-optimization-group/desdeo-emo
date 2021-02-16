"""This module provides implementations of EAs which can be used for training
surrogate models.
"""

__all__ = ["BioGP", "EvoNN", "EvoDN2", "surrogateProblem"]

from desdeo_emo.surrogatemodels.BioGP import BioGP
from desdeo_emo.surrogatemodels.EvoNN import EvoNN
from desdeo_emo.surrogatemodels.EvoDN2 import EvoDN2
from desdeo_emo.surrogatemodels.Problem import surrogateProblem
