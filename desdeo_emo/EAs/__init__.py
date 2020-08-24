"""This module contains classes implementing different Evolutionary algorithms.
"""

__all__ = ["RVEA", "NSGAIII", "BaseEA", "BaseDecompositionEA", "PPGA", "TournamentEA"]

from desdeo_emo.EAs.BaseEA import BaseEA, BaseDecompositionEA
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.PPGA import PPGA
from desdeo_emo.EAs.TournamentEA import TournamentEA
