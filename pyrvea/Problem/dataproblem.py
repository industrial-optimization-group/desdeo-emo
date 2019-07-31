from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from pyrvea.Problem.evonn_problem import EvoNNModel as EvoNN
from pyrvea.Problem.evodn2_problem import EvoDN2Model as EvoDN2
from pyrvea.Problem.biogp_problem import BioGPModel as BioGP
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.metrics import r2_score
from pyrvea.Problem.baseproblem import BaseProblem
import pandas as pd
from typing import List


class DataProblem(BaseProblem):
    def __init__(
        self,
        data: pd.DataFrame = None,
        x: List[str] = None,
        y: List[str] = None,
        minimize: List[bool] = None,
        ideal: List[float] = None,
        nadir: List[float] = None,
        num_of_constraints=0,
        lower_limits: List[float] = None,
        upper_limits: List[float] = None,
        name="data_problem",
    ):
        self.raw_data = data
        self.data = data
        self.x = x
        self.y = y
        self.num_of_variables = len(x)
        self.num_of_objectives = len(y)
        self.num_of_constraints = num_of_constraints
        self.number_of_samples = self.data.shape[0]
        self.name = name
        if minimize is None:
            self.minimize = [True] * self.num_of_objectives
        else:
            assert len(minimize) == self.num_of_objectives
            self.minimize = minimize
        self.preprocessing_transformations = []
        # These indices define how the training will occur.
        # all_indices is the list of indices that can be used. This does not include
        # outliers.
        self.all_indices = list(range(self.number_of_samples))
        # These should be list of lists.
        # The inner list contains the list of indices.
        # The outer list, if it contains multiple elements, should lead to multiple
        # trained models. Can be used for k-fold validation.
        # Train and test indices super list should have the same length.
        self.train_indices: List[List] = []
        self.test_indices: List[List] = []
        self.validation_indices: List[List] = None

        # self.models = dict.fromkeys(self.y, [])  # This method duplicates models in each key, why?
        self.models = {}                           # This works..
        for x in self.y:
            self.models[x] = []

        self.metrics = []
        # Defining bounds in the decision space
        if lower_limits is None:
            self.lower_limits = np.min(self.raw_data[x], axis=0)
        else:
            assert len(lower_limits) == self.num_of_variables
            self.lower_limits = lower_limits
        if upper_limits is None:
            self.upper_limits = np.max(self.raw_data[x], axis=0)
        else:
            assert len(upper_limits) == self.num_of_variables
            self.upper_limits = upper_limits

    def data_scaling(self, data_decision):  # Scales the data from 0 to 1
        # Check this range stuff
        min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=(self.lower_limits, self.upper_limits)
        )
        self.processed_data_decision = min_max_scaler.fit_transform(data_decision)
        self.preprocessing_transformations.append(min_max_scaler)

    def data_uniform_mapping(self):  # Maps the data to uniform distribution
        pass

    def outlier_removal(self):  # Removes the outliers
        pass

    def train_test_split(self, train_size: float = 0.8):  # Split dataset

        for x in range(1):
            train_indices, test_indices = tts(self.all_indices, train_size=train_size)
            train_indices.sort()
            test_indices.sort()
            self.train_indices.append(train_indices)
            self.test_indices.append(test_indices)

    def train(self, model_type: str = None, objectives: str = None, **kwargs):
        if objectives is None:
            objectives = self.y
        if model_type is None:
            model_type = "MLP"
        surrogate_model_options = {
            "GPR": GaussianProcessRegressor,
            "MLP": MLPRegressor,
            "EvoNN": EvoNN,
            "EvoDN2": EvoDN2,
            "BioGP": BioGP
        }
        model_type = surrogate_model_options[model_type]
        # Build specific surrogate models
        print("Building Surrogate Models ...")
        # Fit to data using Maximum Likelihood Estimation of the parameters
        for obj in objectives:

            print("Building model for " + str(obj))
            for train_run, train_indices in enumerate(self.train_indices):
                print("Training run number", train_run, "of", len(self.train_indices))
                model = model_type(**kwargs)
                model.fit(
                    self.data[self.x].loc[train_indices],
                    self.data[obj].loc[train_indices],
                )
                self.models[obj].append(model)

        # Select model
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
        for obj, i in zip(self.y, range(self.num_of_objectives)):
            y = self.models[obj][0].predict(
                decision_variables_transformed.reshape(1, self.num_of_variables),
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

    def objectives(self, decision_variables):
        """Objectives function to use in optimization.

        Parameters
        ----------
        decision_variables : ndarray
            The decision variables

        Returns
        -------
        objectives : ndarray
            The objective values

        """
        objectives = []
        for obj in self.y:
            objectives.append(
                self.models[obj][0].predict(decision_variables.reshape(1, -1))[0]
            )

        return objectives

