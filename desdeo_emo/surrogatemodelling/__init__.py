"""This module provides implementations of EAs which can be used for training
surrogate models.
"""

__all__ = ["BioGP", "EvoNN", "EvoNNforDESDEO", "EvoDN2", "surrogateProblem"]

from desdeo_emo.surrogatemodelling.BioGP import BioGP
from desdeo_emo.surrogatemodelling.EvoNN import EvoNN, EvoNNforDESDEO
from desdeo_emo.surrogatemodelling.EvoDN2 import EvoDN2
from desdeo_emo.surrogatemodelling.Problem import surrogateProblem
