from desdeo_emo.surrogatemodelling.EvoNN import EvoNNforDESDEO
from desdeo_emo.surrogatemodelling.BioGP import BioGP
from desdeo_emo.surrogatemodelling.EvoDN2 import EvoDN2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_problem.surrogatemodels.lipschitzian import LipschitzianRegressor

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs

from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from pygmo import non_dominated_front_2d as nd2

X = np.random.rand(100, 3)
y = X[:, 0] * X[:, 1] + X[:, 2]
y = y.reshape(-1, 1)
data = pd.DataFrame(np.hstack((X, y)), columns=["x1", "x2", "x3", "y"])


problem_name = "ZDT1"
prob = test_problem_builder(problem_name)


x = lhs(30, 100)
y = prob.evaluate(x)

x_names = [f'x{i}' for i in range(1,31)]
y_names = ["f1", "f2"]

data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)

problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)

problem.train(EvoNNforDESDEO, model_parameters={"training_algorithm":RVEA, "pop_size": 50})
#problem.train(EvoNN)
evolver = NSGAIII(problem, use_surrogates=True)
while evolver.continue_evolution():
    evolver.iterate()