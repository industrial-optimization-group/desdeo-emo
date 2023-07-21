import pytest
import numpy as np
from desdeo_emo.EAs.PBEA import PBEA
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.population.Population import Population
from desdeo_problem.testproblems import TestProblems
import pandas as pd
from desdeo_tools.utilities import distance_to_reference_point

#todo
@pytest.fixture
def problem():
    problem = TestProblems.test_problem_builder(name="DTLZ1", n_of_variables=200, n_of_objectives=2)
    return problem

@pytest.fixture
def get_approximation(problem):
    problem = problem
    evolver = IBEA(problem, population_size=32, n_iterations=10, n_gen_per_iter=100,total_function_evaluations=3000)

    while evolver.continue_evolution():
        evolver.iterate()
    individuals, objective_values = evolver.end()
    ini_pop = evolver.population

    return evolver.population.problem.ideal, ini_pop

@pytest.mark.skip(reason="TypeError: 'numpy.ndarray' object is not callable")
def test_PBEA_interaction(problem, get_approximation):
    problem = problem
    approx, ini_pop = get_approximation

    delta = 0.05
    evolver = PBEA(problem, interact=True, population_size=32, initial_population=ini_pop, 
                n_iterations=10, n_gen_per_iter=100, total_function_evaluations=2000, 
                indicator=approx, delta=delta)
    
    # Check that init worked
    assert evolver is not None

    # Check giving preference information
    pref, plot = evolver.requests()
    message = pref.content['message']

    assert message is not None

    response = evolver.population.ideal_fitness_val + [0.2,0.5]
    pref.response = pd.DataFrame([response], columns=pref.content['dimensions_data'].columns)
    pref, plot = evolver.iterate(pref)
    individuals, solutions, _ = evolver.end()

    assert individuals.shape[0] > 0
    assert solutions.shape[0] > 0









