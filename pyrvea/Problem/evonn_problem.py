from math import log
import numpy as np
#from scipy.linalg import lstsq
import timeit
from pyrvea.Problem.baseProblem import baseProblem

class EvoNNProblem():
    """Creates an Artificial Neural Network (ANN) for the EvoNN algorithm."""

    def __init__(
        self,
        training_data_input=None,
        training_data_output=None,
        name=None,
        num_input_nodes=4,
        num_hidden_nodes=5,
        num_of_constraints=0,
        num_of_objectives=2,
        w_low=-5.0,
        w_high=5.0,
    ):

        """

        Parameters
        ----------
        training_data_input : numpy array
            Training data input matrix
        training_data_output : numpy array
            Training data expected output
        name : str
            Name of the sample
        num_input_nodes : int
            The number of nodes in the input layer
        num_hidden_nodes : int
            The number of nodes in the hidden layer
        num_output_nodes : int
            The number of nodes in the output layer
        w_low : float
            The lower bound for randomly generated weights
        w_high : float
            The upper bound for randomly generated weights
        """

        self.training_data_input = training_data_input
        self.training_data_output = training_data_output
        self.name = name
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_of_objectives = num_of_objectives
        self.num_of_variables = training_data_input.shape[1]
        self.num_of_constraints = num_of_constraints
        self.w_low = w_low
        self.w_high = w_high
        self.lower_limits = w_low
        self.upper_limits = w_high
        self.prob_omit = 0.2
        self.bias = 1
        self.num_of_samples = training_data_output.shape[0]

        if len(self.training_data_input) > 0:
            self.num_input_nodes = np.shape(training_data_input)[1]

    def objectives(self, decision_variables) -> list:

        """ Use this method to calculate objective functions.

        Parameters
        ----------
        decision_variables : tuple
            Variables from the neural network

        Returns
        -------
        obj_func : array
            The objective function

        """

        weighted_input = self.dot_product(decision_variables)
        activated_function = self.activation(weighted_input, name="sigmoid")
        w_matrix2, rss, predicted_values = self.minimize_error(activated_function)
        training_error = self.loss_function(predicted_values)

        complexity = self.calculate_complexity(decision_variables)
        obj_func = [training_error, complexity]

        return obj_func

    def dot_product(self, w_matrix):
        """ Calculate the dot product of input and weight + bias.

        Parameters
        ----------
        w_matrix

        Returns
        -------

        """

        # Calculate dot product
        wi = np.dot(self.training_data_input, w_matrix[1:, :]) + w_matrix[0]

        return wi

    def activation(self, wi, name="sigmoid"):
        """ Activation function

        Returns
        -------
        The penultimate layer Z before the output

        """

        if name == "sigmoid":
            activated_function = lambda x: 1 / (1 + np.exp(-x))

        if name == "relu":
            activated_function = lambda x: np.maximum(x, 0)

        return activated_function(wi)

    def minimize_error(self, activated_function, name="llsq"):
        """ Optimize the training error.

        Parameters
        ----------
        activated_function
            Output of the activation function

        Returns
        -------
        """
        if name == "llsq":
            w_matrix2 = np.linalg.lstsq(activated_function, self.training_data_output, rcond=None)
            rss = w_matrix2[1]
            predicted_values = np.dot(activated_function, w_matrix2[0])

        return w_matrix2[0], rss, predicted_values

    def loss_function(self, predicted_values, name="rmse"):

        if name == "rmse":
            return np.sqrt(((self.training_data_output - predicted_values) ** 2).mean())

    def calculate_complexity(self, w_matrix):

        k = np.count_nonzero(w_matrix[1:, :])

        return k

    def information_criterion(self, w_matrix):

        wi = self.dot_product(w_matrix)
        z = self.activation(wi)
        w_matrix2, rss, prediction = self.minimize_error(z)
        #rss = ((self.training_data_output - prediction) ** 2).sum()
        k = self.calculate_complexity(w_matrix) + np.count_nonzero(w_matrix2)

        aic = 2 * k + self.num_of_samples * log(rss/self.num_of_samples)
        aicc = aic + (2*k*(k+1)/(self.num_of_samples-k-1))

        return aicc

    def get_prediction(self, decision_variables):

        weighted_input = self.dot_product(decision_variables)
        activated_function = self.activation(weighted_input)
        w_matrix2, rss, predicted_values = self.minimize_error(activated_function)

        return predicted_values
