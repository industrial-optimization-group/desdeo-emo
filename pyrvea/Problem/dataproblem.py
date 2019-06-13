from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.metrics import r2_score
from pyrvea.Problem.baseProblem import baseProblem
import pandas as pd
from typing import List


class DataProblem(baseProblem):
    def __init__(
        self,
        data: pd.DataFrame = None,
        x: List[str] = None,
        y: List[str] = None,
        minimize: List[bool] = None,
        ideal: List[float] = None,
        nadir: List[float] = None,
        lower_bound: List[float] = None,
        upper_bound: List[float] = None,
    ):
        self.raw_data = data
        self.data = data
        self.x = x
        self.y = y
        self.number_of_variables = len(x)
        self.number_of_objectives = len(y)
        self.number_of_samples = self.data.shape[0]
        if minimize is None:
            self.minimize = [True] * self.number_of_objectives
        else:
            assert len(minimize) == self.number_of_objectives
            self.minimize = minimize
        self.preprocessing_transformations = []
        self.number_of_objectives = len(y)
        # These indices define how the training will occur.
        # all_indices is the list of indices that can be used. This does not include
        # outliers.
        self.all_indices = list(range(self.number_of_samples))
        # These should be list of lists.
        # The inner list contains the list of indices.
        # The outer list, if it contains multiple elements, should lead to multiple
        # trained models. Can be used for k-fold validation.
        # Train and test indices super list should have the same length.
        self.train_indices: List[List] = None
        self.test_indices: List[List] = None
        self.validation_indices: List[List] = None

        self.models = dict.fromkeys(self.y, [])
        self.metrics = []
        # Defining bounds in the decision space
        if lower_bound is None:
            self.lower_bound = np.min(self.raw_data[x], axis=0)
        else:
            assert len(lower_bound) == self.num_of_variables
            self.lower_bound = lower_bound
        if upper_bound is None:
            self.upper_bound = np.max(self.raw_data[x], axis=0)
        else:
            assert len(upper_bound) == self.num_of_variables
            self.upper_bound = upper_bound

    def data_scaling(self, data_decision):  # Scales the data from 0 to 1
        # Check this range stuff
        min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=(self.lower_bound, self.upper_bound)
        )
        self.processed_data_decision = min_max_scaler.fit_transform(data_decision)
        self.preprocessing_transformations.append(min_max_scaler)

    def data_uniform_mapping(self):  # Maps the data to uniform distribution
        pass

    def outlier_removal(self):  # Removes the outliers
        pass

    def train_test_split(self, train_percent: float = 0.7):  # Split dataset
        train_indices, test_indices = tts(self.all_indices, train_percent)

    def train(self, model_type: str = None, objectives: str = None, **kwargs):
        if objectives is None:
            objectives = self.y
        if model_type is None:
            model_type = "GPR"
        surrogate_model_options = {"GPR": GaussianProcessRegressor}
        model_type = surrogate_model_options[model_type]
        # Build specific surrogate models
        print("Building Surrogate Models ...")
        # Fit to data using Maximum Likelihood Estimation of the parameters
        for obj in objectives:
            print("Building model for " + obj)
            for train_indices, train_run in enumerate(self.train_indices):
                print("Training run number", train_run, 'of', len(self.train_indices))
                model = model_type(**kwargs)
                model.fit(
                    self.data[self.x][train_indices], self.data[self.y][train_indices]
                )
                self.models[obj] = self.models[obj].append(model)
        print("Surrogate models build completed.")

    def transform_new_data(self, decision_variables):
        decision_variables_transformed = decision_variables
        if len(self.preprocessing_transformations) > 0:
            for transformation in self.preprocessing_transformations:
                decision_variables_transformed = transformation.transfrom(
                    decision_variables_transformed
                )
        return decision_variables_transformed

    def surrogates_predict(self, decision_variables):
        y_pred = None
        # transforms applied
        decision_variables_transformed = self.transform_new_data(decision_variables)
        for obj, i in zip(self.obj_names, range(self.number_of_objectives)):
            y = self.models[obj].predict(
                decision_variables_transformed.reshape(1, self.number_of_variables),
                return_std=False,
            )
            if y_pred is None:
                y_pred = np.asarray(y)
            else:
                y_pred = np.hstack((y_pred, y))
        return y_pred

    def testing_score(self):  # Return R-squared of testing
        x, y = self.select_data(self.test_indices)
        x = self.transform_new_data(x)
        y_pred = None
        for i in range(np.shape(x)[0]):
            if y_pred is None:
                y_pred = np.asarray(self.surrogates_predict(x[i]))
            else:
                y_pred = np.vstack((y_pred, self.surrogates_predict(x[i])))
        for i in range(np.shape(y)[1]):
            self.r_sq.append(r2_score(y[:, i], y_pred[:, i]))
        return y_pred

    def retrain_surrogate(self):
        pass