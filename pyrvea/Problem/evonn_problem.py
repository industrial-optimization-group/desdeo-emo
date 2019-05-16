import numpy as np


class EvoNNProblem:
    """Creates an Artificial Neural Network (ANN) for the EvoNN algorithm."""

    def __init__(
        self,
        data_input=None,
        data_output=None,
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
        data_input : numpy array
            Training data input matrix
        data_output : numpy array
            Training data desired output
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

        self.data_input = data_input
        self.data_output = data_output
        self.name = name
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.w_low = w_low
        self.w_high = w_high

        if len(self.data_input) > 0:
            self.num_input_nodes = np.shape(data_input)[1]

        # self.w_matrix = np.hstack((w_matrix, bias))

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
                size=(self.num_input_nodes, self.num_hidden_nodes),
            )

            # Initialize bias with a default value of 1

            bias = np.full((self.num_hidden_nodes, 1), 1)

            return w_matrix, bias

        def feed_forward(w_matrix, bias):

            print(np.shape(w_matrix))
            print(np.shape(self.data_input))
            wi = np.dot(self.data_input, w_matrix) + bias

            return wi

        def activation(weighted_input, a_func):
            """ Activation function

            Returns
            -------
            The penultimate layer Z before the output

            """

            def sigmoid(s):
                return 1 / (1 + np.exp(-s))

            if a_func == "sigmoid":
                hidden_layer = sigmoid(weighted_input)  # activation function

            return hidden_layer

        def minimize_error(activated_function):
            """ Optimize the training error

            Parameters
            ----------
            activated_function
                Output of the activation function

            Returns
            -------
            """

            # llsq()
            pass

        def minimize_complexity():
            pass

        w_matrix, bias = init_weight_matrix()
        weighted_input = feed_forward(w_matrix, bias)
        hidden_layer = activation(weighted_input, "sigmoid")
        predicted_output = minimize_error(hidden_layer)

        return self.obj_func(decision_variables)
