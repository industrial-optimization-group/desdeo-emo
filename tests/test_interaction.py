import time
import pytest
import numpy as np
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.MOEAD import MOEA_D
from desdeo_emo.EAs.PBEA import PBEA
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_emo.population.Population import Population
from desdeo_problem.testproblems import TestProblems
import pandas as pd
#from desdeo_problem.testproblems.TestProblems import test_problem_builder

"""
Tests for:
{'Non-preferred solutions': 'Choose one or more solutions that are not '
                            'preferred. The reference vectors near such '
                            'solutions are removed. New solutions are hence '
                            'not searched for in areas close to these '
                            'solutions.',
 'Preferred ranges': 'Provide preferred values for the upper and lower bounds '
                     'of all objectives. New reference vectors are generated '
                     'within these bounds. New solutions are searched for in '
                     'this bounded region of interest.',
 'Preferred solutions': 'Choose one or more solutions as the preferred '
                        'solutions. The reference vectors are focused around '
                        'the vector joining the utopian point and the '
                        'preferred solutions. New solutions are searched for '
                        'in this focused regions of interest.',
 'Reference point': 'Specify a reference point worse than the utopian point. '
                    'The reference vectors are focused around the vector '
                    'joining provided reference point and the utopian point. '
                    'New solutions are searched for in this focused region of '
                    'interest.'}
"""

#Need to add other EAs and then interaction tests are also done.

@pytest.fixture
def problem():
    problem = TestProblems.test_problem_builder(name="DTLZ1", n_of_variables=30, n_of_objectives=2)
    return problem


@pytest.fixture
def create_evolvers(problem):
    evolver_RVEA = RVEA(problem, interact=True, n_iterations=5, n_gen_per_iter=100)
    evolver_NSGAIII = NSGAIII(problem, interact=True, n_iterations=5, n_gen_per_iter=100)

    #evolver_MOEAD = MOEA_D(problem, interact=True, n_iterations=5, n_gen_per_iter=100)  #  non preferred solutions index problem
    #evolver_ibea = IBEA(problem, interact=True, population_size=32, n_iterations=10, n_gen_per_iter=100) # not working no interaction?
    #evolver_PBEA = PBEA(problem, interact=True, n_iterations=5, n_gen_per_iter=100)  #not working yet no interaction?

    evolvers = [evolver_RVEA, evolver_NSGAIII]

    return evolvers


def test_reference_point(create_evolvers):
    evolvers = create_evolvers
    for evolver in evolvers:
        print("testing reference_point for ", evolver)
        evolver.set_interaction_type('Reference point')
        pref, plot = evolver.start()

        #test message
        message = pref.content['message']

        #test response
        response =  evolver.population.ideal_fitness_val + [0.5,0.7]
        pref.response = pd.DataFrame([response], columns=pref.content['dimensions_data'].columns)

        pref, plot = evolver.iterate(pref)
        individuals, solutions, _ = evolver.end()

        # Test that the message and response are not None
        assert message is not None
        #assert pref.response is not None

        # Test that individuals and solutions spaces are not 0
        assert individuals.shape[0] > 0
        assert solutions.shape[0] > 0

def test_preferred_solutions(create_evolvers):
    evolvers = create_evolvers
    for evolver in evolvers:
        print("testing preferred_solutions for ", evolver)
        evolver.set_interaction_type("Preferred solutions")
        pref, plot = evolver.start()

        # test message
        message = pref.content['message']

        # test response
        response = np.array([0, 2])
        pref.response = response

        pref, plot = evolver.iterate(pref)
        individuals, solutions, _ = evolver.end()

        # Test that the message and response are not None
        assert message is not None
        #assert pref.response is not None

        # Test that individuals and solutions spaces are not 0
        assert individuals.shape[0] > 0
        assert solutions.shape[0] > 0

def test_non_preferred_solutions(create_evolvers):
    evolvers = create_evolvers
    
    for evolver in evolvers:
        print("testing non_preferred_solutions for ", evolver)
        evolver.set_interaction_type('Non-preferred solutions')
        pref, plot = evolver.start()

        # test message
        message = pref.content['message']

        # test response
        response = np.array([1, 2], dtype=int)
        #response = np.ravel(response)  # Convert to 1-dimensional array
        pref.response = response

        pref, plot = evolver.iterate(pref)
        individuals, solutions, _ = evolver.end()

        assert message is not None

        # Test that individuals and solutions spaces are not empty
        assert individuals.shape[0] > 0
        assert solutions.shape[0] > 0

        # Test that the non-preferred solutions are not present in the solutions
        assert np.all(np.isin(pref.response, np.arange(individuals.shape[0]))) == False






