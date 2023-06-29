import time
import pytest
import numpy as np
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.population.Population import Population
from desdeo_problem.testproblems import TestProblems
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

@pytest.fixture
def problem():
    problem = TestProblems.test_problem_builder(name="DTLZ1", n_of_variables=30, n_of_objectives=3)
    return problem


@pytest.fixture
def create_evolvers(problem):
    evolver_RVEA = RVEA(problem, interact=True, n_iterations=5, n_gen_per_iter=400)
    evolver_NSGAIII = NSGAIII(problem, interact=True, n_iterations=5, n_gen_per_iter=400)

    evolvers = [evolver_RVEA, evolver_NSGAIII]

    return evolvers

def test_reference_point(create_evolvers):
    evolvers = create_evolvers
    evolver1 = evolvers[0]

    for evolver in evolvers:
        evolver.set_interaction_type('Reference point')
        pref, plot = evolver.start()
        message = pref.content['message']
        response = [0., 0., 0.]  # Use [0, 0, 0] as the reference point
        pref = None 
        pref, plot = evolver1.iterate(pref)
        individuals, solutions, _ = evolver1.end()

        # Test that the message and response are not None
        assert message is not None
        assert response is not None

        # Test that individuals and solutions spaces are not 0
        assert individuals.shape[0] > 0
        assert solutions.shape[0] > 0




