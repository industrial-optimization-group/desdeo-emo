import numpy as np
from scipy.linalg import lstsq

class EvoNNProblem:
    """Creates an Artificial Neural Network (ANN) for the EvoNN algorithm."""

    def __init__(
        self,
        training_data_input=None,
        training_data_output=None,
        name=None,
        num_input_nodes=4,
        num_hidden_nodes=6,
        num_output_nodes=1,
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
        self.num_output_nodes = num_output_nodes
        self.w_low = w_low
        self.w_high = w_high

        if len(self.training_data_input) > 0:
            self.num_input_nodes = np.shape(training_data_input)[1]

    def objectives(self, decision_variables=None) -> list:

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

        def init_weight_matrix():

            # Initialize the weight matrix with random weights within boundaries
            # Rows = hidden nodes
            # Columns = input nodes
            # Last column is for bias

            w_matrix = np.random.uniform(
                self.w_low,
                self.w_high,
                size=(self.num_hidden_nodes, self.num_input_nodes),
            )

            bias = np.full((self.num_hidden_nodes, 1), 1)

            return w_matrix, bias

        def dot_product(w_matrix, bias):
            """ Calculate the dot product of input and weight + bias.
                TODO: %timeit whether it's faster to keep bias in its own
                    vector, or as a column in weight matrix

            Parameters
            ----------
            w_matrix
            bias

            Returns
            -------

            """

            wi = np.dot(self.training_data_input, w_matrix.transpose()) + bias
            # biased_matrix = np.hstack((w_matrix, bias))
            # biased_dot = np.dot(self.data_input, biased_matrix[..., :-1].transpose()) + biased_matrix[:,-1]

            return wi

        def activation(weighted_input, name):
            """ Activation function

            Returns
            -------
            The penultimate layer Z before the output

            """

            def sigmoid(s):
                return 1 / (1 + np.exp(-s))

            if name == "sigmoid":
                activated_function = sigmoid(weighted_input)  # activation function

            return activated_function

        def optimize_error(activated_function, name):
            """ Optimize the training error

            Parameters
            ----------
            activated_function
                Output of the activation function

            Returns
            -------
            """
            if name == "llsq":
                w_matrix2 = lstsq(activated_function, self.training_data_output)
                predicted_values = np.dot(activated_function, w_matrix2)


            return w_matrix2, predicted_values

        def calculate_error(predicted_values, name):

            if name == "rmse":
                return np.sqrt(((self.training_data_output - predicted_values) ** 2).mean())

        def calculate_complexity(w_matrix, w_matrix2):

            k = np.nonzero(w_matrix) + np.nonzero(w_matrix2)

            

        w_matrix, bias = init_weight_matrix()
        weighted_input = dot_product(w_matrix, bias)
        activated_function = activation(weighted_input, "sigmoid")
        w_matrix2, predicted_values = optimize_error(activated_function, "llsq")
        training_error = calculate_error(predicted_values, "rmse")

        complexity = calculate_complexity(w_matrix, w_matrix2)


        return self.obj_func(training_error)
