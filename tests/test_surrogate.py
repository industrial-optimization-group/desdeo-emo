import numpy as np
import pandas as pd
import pytest
from desdeo_problem import DataProblem
from desdeo_emo.surrogatemodels.EvoNN import EvoNN
from desdeo_emo.surrogatemodels.EvoDN2 import EvoDN2
from desdeo_emo.surrogatemodels.BioGP import BioGP
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs import NSGAIII, PPGA
import matplotlib.pyplot as plt

@pytest.fixture
def create_data():
    X = np.random.rand(100, 3)
    y = X[:, 0] * X[:, 1] + X[:, 2]
    y = y.reshape(-1, 1)
    data = pd.DataFrame(np.hstack((X, y)), columns=["x1", "x2", "x3", "y"])
    return X, y, data


# These do not work proprely yet
"""
def test_EvoNN(create_data):
    X, y, data = create_data
    model = EvoNN(pop_size=100)
    model.fit(data[["x1", "x2", "x3"]], data['y'])
    y_pred = model.predict(X)
"""

"""
def test_EvoDN2(create_data):
    X, y, data = create_data
    model3 = EvoDN2(training_algorithm=NSGAIII, pop_size=50)
    model3.fit(data[["x1", "x2", "x3"]], data['y'])
    model3.predict(X)
"""

"""
def test_bioGP(create_data):
    X, y, data = create_data
    model2 = BioGP(pop_size=50)
    model2.fit(data[["x1", "x2", "x3"]], data['y'])
    X_pred = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    y_pred, _ = model2.predict(X_pred)
"""


