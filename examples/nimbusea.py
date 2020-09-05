import numpy as np
import pandas as pd

from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs.NIMBUS_RVEA import NIMBUS_RVEA

problem = test_problem_builder(name="ZDT1")
problem.ideal = np.asarray([0, 0])
problem.nadir = np.asarray([1, 1])

evolver = NIMBUS_RVEA(problem, n_gen_per_iter=10)

plot, pref = evolver.requests()

pref.response = pd.DataFrame([[0.1, 0.9]], columns=pref.content['dimensions_data'].columns)

plot, pref = evolver.iterate(pref)