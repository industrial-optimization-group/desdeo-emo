from math import ceil
from typing import Callable, Dict, Type, Union
import random


import numpy as np
import pandas as pd
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from scipy.special import expit
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

from desdeo_emo.EAs.BaseEA import BaseEA
from desdeo_emo.EAs.PPGA import PPGA

from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
from desdeo_emo.recombination.evodn2_xover_mutation import EvoDN2Recombination
from desdeo_emo.surrogatemodelling.Problem import surrogateProblem


def negative_r2_score(y_true, y_pred):
    return -r2_score(y_true, y_pred)


class EvoDN2(BaseRegressor):
    def __init__(
        self,
        num_subnets: int = 4,
        num_subsets: int = 4,
        max_layers: int = 4,
        max_nodes: int = 4,
        p_omit: float = 0.2,
        w_low: float = -5.0,
        w_high: float = 5.0,
        subsets: list = None,
        activation_function: str = "sigmoid",
        loss_function: str = "mse",
        training_algorithm: BaseEA = PPGA,
        pop_size: int = 500,
        model_selection_criterion: str = "min_error",
        verbose: int = 0,
    ):
        loss_functions = {
            "mse": mean_squared_error,
            "msle": mean_squared_log_error,
            "neg_r2": negative_r2_score,
        }
        # Model Hyperparameters
        self.num_subnets: int = num_subnets
        self.num_subsets: int = num_subsets
        self.max_layers: int = max_layers
        self.max_nodes: int = max_nodes
        self.p_omit: float = p_omit
        self.w_low: float = w_low
        self.w_high: float = w_high
        self.subsets: list = subsets
        self.activation_function: str = activation_function
        self.loss_function_str: str = loss_function
        self.loss_function: Callable = loss_functions[loss_function]
        # EA Parameters
        self.training_algorithm: BaseEA = training_algorithm
        self.pop_size: int = pop_size
        self.model_selection_criterion: str = model_selection_criterion
        self.verbose: int = verbose
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.model_trained: bool = False
        # Model parameters
        self._subnets = None
        self._last_layer = None
        # Extras
        self.performance: Dict = {"RMSE": None, "R^2": None, "Complexity": None}
        self.model_population = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            msg = (
                f"Ensure that the number of samples in X and y are the same"
                f"Number of samples in X = {X.shape[0]}"
                f"Number of samples in y = {y.shape[0]}"
            )
            raise ModelError(msg)
        self.X = X
        self.y = y
        if self.subsets is None:
            self.subsets = []

            # Create random subsets of decision variables for each subnet

            for i in range(self.num_subnets):
                n = random.randint(1, self.X.shape[1])
                self.subsets.append(random.sample(range(self.X.shape[1]), n))

            # Ensure that each decision variable is used as an input in at least one subnet
            for n in list(range(self.X.shape[1])):
                if not any(n in k for k in self.subsets):
                    self.subsets[random.randint(0, self.num_subnets - 1)].append(n)

        # Create problem
        problem = surrogateProblem(performance_evaluator=self._model_performance)
        problem.n_of_objectives = 2
        # Create Population
        initial_pop = self._create_individuals()
        population = SurrogatePopulation(
            problem, self.pop_size, initial_pop, None, None, None
        )
        # Do evolution
        evolver = self.training_algorithm(problem, initial_population=population)
        recombinator = EvoDN2Recombination(evolver=evolver)
        evolver.population.recombination = recombinator
        figure = animate_init_(evolver.population.objectives, filename="EvoDN2.html")
        while evolver.continue_evolution():
            evolver.iterate()
            figure = animate_next_(
                evolver.population.objectives,
                figure,
                filename="EvoDN2.html",
                generation=evolver._iteration_counter,
            )
        self.model_population = evolver.population
        # Selection
        self.select()
        self.model_trained = True

    def _model_performance(
        self,
        individuals: np.ndarray = None,
        X: np.ndarray = None,
        y_true: np.ndarray = None,
    ):
        if individuals is None and self.model_trained is False:
            msg = "Model has not been trained yet"
            raise ModelError(msg)
        if individuals is None:
            individuals = self._subnets
        if X is None:
            X = self.X
            y = self.y
        if len(individuals) > 1:
            loss = []
            complexity = []
            for individual in individuals:
                penultimate_y_pred, ind_complexity = self._feed_forward(individual, X)
                y_pred, _ = self._calculate_linear(penultimate_y_pred)
                loss.append(self.loss_function(y, y_pred))
                complexity.append(ind_complexity)
        else:
            penultimate_y_pred, ind_complexity = self._feed_forward(individuals, X)
            y_pred, _ = self._calculate_linear(penultimate_y_pred)
            loss = self.loss_function(y, y_pred)
            complexity = ind_complexity
        return np.asarray((loss, complexity)).T

    def _feed_forward(self, subnets, X):
        network_complexity = []
        penultimate_output = np.empty((X.shape[0], 0))
        for i, subnet in enumerate(subnets):
            # Get the input variables for the first layer
            feed = X[:, self.subsets[i]]
            subnet_complexity = 1
            for layer in subnet:
                # Calculate the dot product + bias
                out = np.dot(feed, layer[1:, :]) + layer[0]
                subnet_complexity = np.dot(subnet_complexity, np.abs(layer[1:, :]))
                feed = self.activate(out)
            network_complexity.append(np.sum(subnet_complexity))
            penultimate_output = np.hstack((penultimate_output, feed))

        complexity = np.sum(network_complexity)
        return penultimate_output, complexity

    def _calculate_linear(self, previous_layer_output):
        """ Calculate the final layer using LLSQ or

        Parameters
        ----------
        non_linear_layer : np.ndarray
            Output of the activation function

        Returns
        -------
        linear_layer : np.ndarray
            The optimized weight matrix of the upper part of the network
        predicted_values : np.ndarray
            The prediction of the model
        """

        linear_layer = None
        previous_layer_output = np.hstack(
            (np.ones((previous_layer_output.shape[0], 1)), previous_layer_output)
        )
        linear_solution = np.linalg.lstsq(previous_layer_output, self.y, rcond=None)
        linear_layer = linear_solution[0]
        y_pred = np.dot(previous_layer_output, linear_layer)
        return y_pred, linear_layer

    def activate(self, x):
        if self.activation_function == "sigmoid":
            return expit(x)
        elif self.activation_function == "relu":
            return np.maximum(x, 0)
        elif self.activation_function == "tanh":
            return np.tanh(x)
        else:
            msg = (
                f"Given activation function not recognized: {self.activation_function}"
                f"\nActivation function should be one of ['relu', 'sigmoid', 'tanh']"
            )
            raise ModelError(msg)

    def predict(self, X):
        penultimate_y_pred, _ = self._feed_forward(self.subnets, X)
        y_pred = (
            np.dot(penultimate_y_pred, self._last_layer[1:, :]) + self._last_layer[0]
        )
        return y_pred

    def select(self):
        if self.model_selection_criterion == "min_error":
            # Return the model with the lowest error
            selected = np.argmin(self.model_population.objectives[:, 0])
        else:
            raise ModelError("Selection criterion not recognized. Use 'min_error'.")
        self.subnets = self.model_population.individuals[selected]
        penultimate_y_pred, complexity = self._feed_forward(self.subnets, self.X)
        y_pred, linear_layer = self._calculate_linear(penultimate_y_pred)
        self._last_layer = linear_layer
        self.performance["RMSE"] = np.sqrt(mean_squared_error(self.y, y_pred))
        self.performance["R^2"] = r2_score(self.y, y_pred)
        self.performance["Complexity"] = complexity

    def _create_individuals(self):
        individuals = []
        for i in range(self.pop_size):
            nets = []
            for j in range(self.num_subnets):

                layers = []
                num_layers = np.random.randint(1, self.max_layers)
                in_nodes = len(self.subsets[j])

                for k in range(num_layers):
                    out_nodes = random.randint(2, self.max_nodes)
                    net = np.random.uniform(
                        self.w_low,
                        self.w_high,
                        size=(in_nodes, out_nodes),
                    )
                    # Randomly set some weights to zero
                    zeros = np.random.choice(
                        np.arange(net.size),
                        ceil(net.size * self.p_omit),
                    )
                    net.ravel()[zeros] = 0

                    # Add bias
                    net = np.insert(net, 0, 1, axis=0)
                    in_nodes = out_nodes
                    layers.append(net)

                nets.append(layers)

            individuals.append(nets)

        return np.asarray(individuals)