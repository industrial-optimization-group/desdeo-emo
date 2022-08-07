import numpy as np
import pandas as pd

from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError


class LipschitzianRegressor(BaseRegressor):
    def __init__(self, L: float = None):
        self.L: float = L
        self.X: np.ndarray = None
        self.y: np.ndarray = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self_dist_x = self.self_distance(X)
        self_dist_y = self.self_distance(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.true_divide(self_dist_y, self_dist_x)
            delta[~np.isfinite(delta)] = -np.inf
            L = delta.max()
        self.X = X
        self.y = y
        self.L = L

    def predict(self, X):
        dist = self.distance(X, self.X)
        y_low = (self.y - self.L * dist).max(axis=0)
        y_high = (self.y + self.L * dist).min(axis=0)
        y_mean = (y_low + y_high) / 2
        y_delta = np.abs((y_high - y_low) / 2)
        return (y_mean, y_delta)

    def self_distance(self, arr):
        if arr.ndim == 1:
            dist = np.abs(np.subtract(arr[None, :], arr[:, None]))
        elif arr.ndim == 2:
            dist = np.sum(np.abs(np.subtract(arr[None, :, :], arr[:, None, :])), axis=2)
        else:
            msg = (
                f"Array of wrong dimension. Expected dimension = 1 or 2. Recieved "
                f"dimension = {arr.ndim}"
            )
            raise ModelError(msg)
        return dist

    def distance(self, array1, array2):
        if array1.ndim == 1:
            array1 = array1.reshape(-1, 1)
        if array2.ndim == 1:
            array2 = array2.reshape(-1, 1)
        dist = np.sum(
            np.abs(np.subtract(array1[None, :, :], array2[:, None, :])), axis=2
        )
        return dist
