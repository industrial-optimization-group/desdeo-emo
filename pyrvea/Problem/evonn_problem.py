from pyrvea.Problem.baseProblem import baseProblem
import numpy as np
import plotly
import plotly.graph_objs as go


class EvoNN(baseProblem):
    """Creates an Artificial Neural Network (ANN) for the EvoNN algorithm.

    Attributes
    ----------
    name : str
        Name of the sample
    num_input_nodes : int
        The number of nodes in the input layer
    num_hidden_nodes : int
        The number of nodes in the hidden layer
    num_of_objectives : int
        The number of objectives
    w_low : float
        The lower bound for randomly generated weights
    w_high : float
        The upper bound for randomly generated weights
    prob_omit : float
        The probability of setting some weights to zero initially
    activation_func : str
        The function to use for activating the hidden layer
    opt_func : str
        The function to use for optimizing the upper part of the network
    loss_func : str
        The loss function to use
    """

    def __init__(
        self,
        name,
        training_data_input=None,
        training_data_output=None,
        num_input_nodes=4,
        num_hidden_nodes=5,
        num_of_objectives=2,
        w_low=-5.0,
        w_high=5.0,
        prob_omit=0.2,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse"
    ):
        super().__init__()

        self.name = name
        self.training_data_input = training_data_input
        self.training_data_output = training_data_output
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_of_objectives = num_of_objectives
        self.w_low = w_low
        self.w_high = w_high
        self.prob_omit = prob_omit
        self.activation_func = activation_func
        self.opt_func = opt_func
        self.loss_func = loss_func
        self.trained_models = []

        if training_data_input is not None and training_data_output is not None:
            self.num_of_samples = training_data_output.shape[0]
            self.num_of_variables = training_data_input.shape[1]
            self.num_input_nodes = self.num_of_variables

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.

        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values
        """

        self.training_data_input = training_data
        self.training_data_output = target_values
        self.num_of_samples = target_values.shape[0]
        self.num_of_variables = training_data.shape[1]
        if len(self.training_data_input) > 0:
            self.num_input_nodes = np.shape(training_data)[1]

    def objectives(self, decision_variables) -> list:

        """ Use this method to calculate objective functions.

        Parameters
        ----------
        decision_variables : ndarray
            Variables from the neural network

        Returns
        -------
        obj_func : list
            The objective function

        """

        activated_layer = self.activation(decision_variables)
        _, _, predicted_values = self.minimize_error(activated_layer)
        training_error = self.loss_function(predicted_values)
        complexity = self.calculate_complexity(decision_variables)

        obj_func = [training_error, complexity]

        return obj_func

    def activation(self, decision_variables):
        """ Calculates the dot product and applies the activation function.

        Parameters
        ----------
        decision_variables : ndarray
            Variables from the neural network
        name : str
            The activation function to use

        Returns
        -------
        The penultimate layer Z before the output

        """
        w1 = decision_variables
        # Calculate the dot product
        wi = (
            np.dot(self.training_data_input, w1[1:, :])
            + w1[0]
        )

        if self.activation_func == "sigmoid":
            activated_layer = lambda x: 1 / (1 + np.exp(-x))

        if self.activation_func == "relu":
            activated_layer = lambda x: np.maximum(x, 0)

        if self.activation_func == "tanh":
            activated_layer = lambda x: np.tanh(x)

        return activated_layer(wi)

    def minimize_error(self, activated_layer):
        """ Minimize the training error.

        Parameters
        ----------
        activated_layer : ndarray
            Output of the activation function
        name : str
            Name of the optimizing algorithm to use

        Returns
        -------
        w_matrix[0] : ndarray
            The weight matrix of the upper part of the network
        rss : float
            Sums of residuals
        predicted_values : ndarray
            The prediction of the model
        """

        if self.opt_func == "llsq":
            w2 = np.linalg.lstsq(activated_layer, self.training_data_output, rcond=None)
            rss = w2[1]
            predicted_values = np.dot(activated_layer, w2[0])

        return w2[0], rss, predicted_values

    def loss_function(self, predicted_values):

        if self.loss_func == "mse":
            return ((self.training_data_output - predicted_values) ** 2).mean()
        if self.loss_func == "rmse":
            return np.sqrt(((self.training_data_output - predicted_values) ** 2).mean())

    def calculate_complexity(self, w_matrix):

        k = np.count_nonzero(w_matrix[1:, :])

        return k

    def information_criterion(self, decision_variables):

        z = self.activation(decision_variables)
        w_matrix2, rss, prediction = self.minimize_error(z)
        # rss = ((self.training_data_output - prediction) ** 2).sum()
        k = self.calculate_complexity(decision_variables) + np.count_nonzero(w_matrix2)
        aic = 2 * k + self.num_of_samples * np.log(rss / self.num_of_samples)
        aicc = aic + (2 * k * (k + 1) / (self.num_of_samples - k - 1))

        return aicc

    def select(self, pop, non_dom_front, criterion="min_error"):
        """ Select target model from the population.

        Parameters
        ----------
        pop : obj
            The population object
        non_dom_front : list
            Indices of the models on the non-dominated front
        criterion : str
            The criterion to use for selecting the model.
            Possible values: 'min_error', 'akaike_corrected', 'manual'

        Returns
        -------
        The selected model
        """
        model = None
        if criterion == "min_error":
            # Return the model with the lowest error

            lowest_error = np.argmin(pop.objectives[:, 0])
            model = TrainedModel(name=self.name+"_model", w1=pop.individuals[lowest_error])

        elif criterion == "akaike_corrected":

            # Calculate Akaike information criterion for the non-dominated front
            # and return the model with the lowest value

            info_c_rank = []

            for i in non_dom_front:

                info_c = self.information_criterion(pop.individuals[i])
                info_c_rank.append((info_c, i))

            info_c_rank.sort()

            model = TrainedModel(name=self.name+"_model", w1=pop.individuals[info_c_rank[0][1]])

        self.trained_models.append(model)

        return model


class TrainedModel(EvoNN):

    def __init__(self, name, w1, w2=None, y_pred=None, svr=None):
        super().__init__(name)
        self.name = name
        self.w1 = w1
        self.w2 = w2
        self.y_pred = y_pred
        self.svr = svr

    def init_upper_part(self):

        activated_layer = super().activation(self.w1)
        self.w2, _, self.y_pred = self.minimize_error(activated_layer)

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.

        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values
        """

        self.training_data_input = training_data
        self.training_data_output = target_values
        self.num_of_samples = target_values.shape[0]
        self.num_of_variables = training_data.shape[1]
        if len(self.training_data_input) > 0:
            self.num_input_nodes = np.shape(training_data)[1]

        self.init_upper_part()

    def predict(self, decision_variables):

        activated_layer = self.activation(decision_variables)
        out = np.dot(activated_layer, self.w2)

        return out

    def activation(self, decision_variables):
        """ Calculates the dot product and applies the activation function.

        Parameters
        ----------
        decision_variables : ndarray
            Variables from the neural network
        name : str
            The activation function to use

        Returns
        -------
        The penultimate layer Z before the output

        """
        wi = (
                np.dot(decision_variables, self.w1[1:, :])
                + self.w1[0]
        )

        if self.activation_func == "sigmoid":
            activated_layer = lambda x: 1 / (1 + np.exp(-x))

        if self.activation_func == "relu":
            activated_layer = lambda x: np.maximum(x, 0)

        if self.activation_func == "tanh":
            activated_layer = lambda x: np.tanh(x)

        return activated_layer(wi)

    def single_variable_response(self, ploton=False, log=None):

        trend = np.loadtxt("trend")
        trend = trend[0:self.num_of_samples]
        avg = np.ones((1, self.num_input_nodes))*(0+1)/2
        svr = np.empty((0, 2))

        for i in range(self.num_input_nodes):
            variables = np.ones((len(trend), 1))*avg
            variables[:, i] = trend

            out = self.predict(variables)

            if min(out) == max(out):
                out = 0.5 * np.ones(out.size)
            else:
                out = (out - min(out)) / (max(out) - min(out))

            if ploton:
                trace0 = go.Scatter(x=np.arange(len(variables[:, 1])), y=variables[:, i], name="input")
                trace1 = go.Scatter(x=np.arange(len(variables[:, 1])), y=out, name="output")
                data = [trace0, trace1]
                plotly.offline.plot(
                    data,
                    filename="x"+str(i+1)+"_response.html",
                    auto_open=True,
                )

            p = np.diff(out)
            q = np.diff(trend)
            r = np.multiply(p, q)
            r_max = max(r)
            r_min = min(r)
            if r_max <= 0 and r_min <= 0:
                response = -1
                s = "inverse"
            elif r_max >= 0 and r_min >= 0:
                response = 1
                s = "direct"
            elif r_max == 0 and r_min == 0:
                response = 0
                s = "nil"
            elif r_min < 0 < r_max:
                response = 2
                s = "mixed"

            print("x" + str(i + 1) + " response: " + str(response) + " " + s, file=log)
            svr = np.vstack((svr, ["x"+str(i+1), s]))
            self.svr = svr
