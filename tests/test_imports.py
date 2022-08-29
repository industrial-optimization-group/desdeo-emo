import pytest

# DUMMY Import test


# just to test that all functions etc get imported properly. Then test them.
# these should work
def test_imports():
    from desdeo_emo.EAs.NSGAIII import NSGAIII
    from desdeo_emo.EAs.IBEA import IBEA
    # from desdeo_emo.EAs.PBEA import PBEA # desdeo-tools version
    from desdeo_emo.EAs.IKRVEA import IK_RVEA
    from desdeo_emo.EAs.RVEA import RVEA
    # from desdeo_emo.EAs.slowRVEA import slowRVEA # legacy code
    from desdeo_emo.EAs.IOPIS import BaseIOPISDecompositionEA
    from desdeo_emo.EAs.MOEAD import MOEA_D
    from desdeo_emo.EAs.PPGA import PPGA
    from desdeo_emo.EAs.TournamentEA import TournamentEA
    from desdeo_emo.EAs.BaseEA import BaseEA, BaseDecompositionEA
    from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
    print("importing EAs work")

    from desdeo_emo.population.CreateIndividuals import create_new_individuals
    from desdeo_emo.population.Population import BasePopulation, Population
    from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
    print("importing populations work")

    from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
    from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover
    from desdeo_emo.recombination.SinglePointCrossoverMutation import SinglePoint_Mutation, SinglePoint_Xover
    print("importing recombination work")

    from desdeo_emo.selection.APD_Select import APD_Select
    from desdeo_emo.selection.APD_Select_constraints import APD_Select
    from desdeo_emo.selection.IOPIS_APD import IOPIS_APD_Select
    from desdeo_emo.selection.IOPIS_NSGAIII import IOPIS_NSGAIII_select
    from desdeo_emo.selection.MOEAD_select import MOEAD_select
    from desdeo_emo.selection.NSGAIII_select import NSGAIII_select
    from desdeo_emo.selection.SelectionBase import SelectionBase
    from desdeo_emo.selection.TournamentSelection import TournamentSelection
    from desdeo_emo.selection.oAPD import Optimistic_APD_Select
    from desdeo_emo.selection.robust_APD import robust_APD_Select 
    print("importing selection work")

    from desdeo_emo.surrogatemodels.BioGP import BioGP
    from desdeo_emo.surrogatemodels.EvoDN2 import EvoDN2, EvoDN2Recombination
    from desdeo_emo.surrogatemodels.EvoNN import EvoNN, EvoNNRecombination
    from desdeo_emo.surrogatemodels.Problem import surrogateProblem
    print("importing surrogatemodels work")

    from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

    print("success")


# these fail right now.
def fail_import():
    from desdeo_emo.EAs.PBEA import PBEA
    from desdeo_emo.EAs.slowRVEA import slowRVEA


def test_import_fails():
    with pytest.raises(ImportError):
        fail_import()
