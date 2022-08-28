"""This module contains classes implementing different Evolutionary algorithms.
"""

__all__ = [
    "RVEA",
    "NSGAIII",
    "BaseEA",
    "BaseDecompositionEA",
    "PPGA",
    "TournamentEA",
    "IOPIS_NSGAIII",
    "IOPIS_RVEA",
    "MOEA_D",
    "IBEA"
]

from desdeo_emo.EAs.BaseEA import BaseEA, BaseDecompositionEA
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.PPGA import PPGA
from desdeo_emo.EAs.TournamentEA import TournamentEA
from desdeo_emo.EAs.IOPIS import IOPIS_NSGAIII, IOPIS_RVEA
from desdeo_emo.EAs.MOEAD import MOEA_D
from desdeo_emo.EAs.IBEA import IBEA
