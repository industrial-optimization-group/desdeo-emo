from math import ceil
from typing import Callable, Type, Union

import numpy as np
import pandas as pd
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from scipy.special import expit
from scipy.optimize import lsq_linear

from desdeo_emo.EAs.BaseEA import BaseEA
from desdeo_emo.EAs.PPGA import PPGA
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
from desdeo_emo.surrogatemodelling.Problem import surrogateProblem
from desdeo_emo.recombination.evonn_xover_mutation import EvoNNRecombination
from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_


def negative_r2_score(y_true, y_pred):
    return -r2_score(y_true, y_pred)


class EvoNN(BaseRegressor):
    def __init__(
        self,
        num_hidden_nodes: int = 20,
        p_omit: float = 0.2,
        w_low: float = -5.0,
        w_high: float = 5.0,
        activation_function: Union[str, Callable] = "sigmoid",  # TODO Give default
        optimization_function: str = "llsq",  # TODO Give default
        loss_function: str = "mse",  # TODO Give default
        training_algorithm: Type[BaseEA] = PPGA,
        pop_size: int = 500,
        model_selection_criterion: str = None,  # TODO Give default
        recombination_type: str = "evonn_xover_mutation",
        crossover_type: str = "standard",
        mutation_type: str = "gaussian",
    ):
        loss_functions = {
            "mse": mean_squared_error,
            "msle": mean_squared_log_error,
            "neg_r2": negative_r2_score,
        }
        # Hyperparameters
        self.num_hidden_nodes: int = num_hidden_nodes
        self.p_omit: float = p_omit
        self.w_low: float = w_low
        self.w_high: float = w_high
        self.activation_function: Union[str, Callable] = activation_function
        self.optimization_function: str = optimization_function
        self.loss_function_str: str = loss_function
        self.loss_function: Callable = loss_functions[loss_function]
        self.training_algorithm: Type[BaseEA] = training_algorithm
        self.pop_size: int = pop_size
        self.model_selection_criterion: str = model_selection_criterion
        self.recombination_type: str = recombination_type
        self.crossover_type: str = crossover_type
        self.mutation_type: str = mutation_type
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.model_trained: bool = False
        # Model Parameters
        self._first_layer: np.ndarray = None
        self._last_layer: np.ndarray = None

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
        figure = animate_init_(evolver.population.objectives, filename="evoNN.html")
        recombinator = EvoNNRecombination(
            evolver=evolver, mutation_type=self.mutation_type
        )
        evolver.population.recombination = recombinator
        while evolver.continue_evolution():
            evolver.iterate()
            fitness = evolver.population.fitness
            figure = animate_next_(
                evolver.population.objectives,
                figure,
                filename="evoNN.html",
                generation=evolver._iteration_counter,
            )

        # Save model's last layer
        self.model_trained = True

    def _model_performance(
        self,
        first_layer: np.ndarray = None,
        X: np.ndarray = None,
        y_true: np.ndarray = None,
    ):
        if first_layer is None and self.model_trained is False:
            msg = "Model has not been trained yet"
            raise ModelError(msg)
        if first_layer is None:
            first_layer = self._first_layer
        if X is None:
            X = self.X
            y = self.y
        if first_layer.ndim == 3:
            loss = []
            complexity = []
            for actual_first_layer in first_layer:
                y_predict = self.predict(X=X, first_layer=actual_first_layer)
                loss.append(self.loss_function(y, y_predict))
                complexity.append(np.count_nonzero(actual_first_layer))
        elif first_layer.ndim == 2:
            y_predict = self.predict(X=X, first_layer=first_layer)
            loss = self.loss_function(y, y_predict)
            complexity = np.count_nonzero(first_layer)
        return np.asarray((loss, complexity)).T

    def predict(self, X: np.ndarray = None, first_layer: np.ndarray = None):
        if first_layer is None and self.model_trained is False:
            msg = "Model has not been trained yet"
            raise ModelError(msg)
        elif first_layer is not None:
            # Calculate the dot product + bias
            out = np.dot(X, first_layer[1:, :]) + first_layer[0]
            activated_layer = self.activate(self.activation_function, out)
            _, y_pred = self.calculate_linear(activated_layer)
        elif first_layer is None:
            first_layer = self._first_layer
            # Calculate the dot product + bias
            out = np.dot(X, first_layer[1:, :]) + first_layer[0]
            activated_layer = self.activate(self.activation_function, out)
            y_pred = np.dot(activated_layer, self._last_layer)
        else:
            msg = "How did you get here?"
            raise ModelError(msg)
        return y_pred

    def activate(self, name, x):
        if name == "sigmoid":
            return expit(x)
        if name == "relu":
            return np.maximum(x, 0)
        if name == "tanh":
            return np.tanh(x)

    def calculate_linear(self, non_linear_layer_output):
        """ Apply the linear function to the activated layer.

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
        training_error : float
            The model's training error
        """

        linear_layer = None

        if self.optimization_function == "llsq":
            linear_solution = np.linalg.lstsq(
                non_linear_layer_output, self.y, rcond=None
            )
            linear_layer = linear_solution[0]

        elif self.optimization_function == "llsq_constrained":
            linear_layer = lsq_linear(
                non_linear_layer_output, self.y_train, method="bvls", bounds=(0, 1)
            ).x

        predicted_values = np.dot(non_linear_layer_output, linear_layer)

        return linear_layer, predicted_values

    def _create_individuals(self):

        individuals = np.random.uniform(
            self.w_low,
            self.w_high,
            size=(self.pop_size, self.X.shape[1], self.num_hidden_nodes),
        )

        # Set bias
        individuals = np.insert(individuals, 0, 1, axis=1)

        # Randomly set some weights to zero
        zeros = np.random.choice(
            np.arange(individuals.size), ceil(individuals.size * self.p_omit)
        )
        individuals.ravel()[zeros] = 0
        return individuals
