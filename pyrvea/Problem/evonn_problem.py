from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from math import ceil
from scipy.special import expit
import numpy as np
import plotly
import plotly.graph_objs as go


class EvoNN(baseProblem):
    """Creates Artificial Neural Network (ANN) models for the EvoNN algorithm.

    These models contain only one hidden node layer. The lower part of the network
    is optimized by a genetic algorithm, and the upper part is optimized by Linear Least Square
    algorithm by default.

    Parameters
    ----------
    name : str
        Name of the sample
    X_train : ndarray
        Training data input
    y_train : ndarray
        Training data target values
    num_input_nodes : int
        The number of nodes in the input layer
    num_nodes : int
        The number of nodes in the hidden layer
    num_of_objectives : int
        The number of objectives
    w_low : float
        The lower bound for randomly generated weights
    w_high : float
        The upper bound for randomly generated weights
    params : dict
        Parameters for model training
    """

    def __init__(
        self,
        name,
        X_train=None,
        y_train=None,
        num_input_nodes=4,
        num_nodes=5,
        num_of_objectives=2,
        w_low=-5.0,
        w_high=5.0,
        params=None,
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.num_input_nodes = num_input_nodes
        self.num_nodes = num_nodes
        self.num_of_objectives = num_of_objectives
        self.w_low = w_low
        self.w_high = w_high
        self.params = params

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.

        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values
        """

        self.X_train = training_data
        self.y_train = target_values
        self.num_of_samples = target_values.shape[0]
        self.num_of_variables = training_data.shape[1]
        self.num_input_nodes = self.num_of_variables
        self.num_nodes = self.params["num_nodes"]

    def create_population(self):
        """Create a population of neural networks for the EvoNN problem.

        Individuals are 2d arrays representing the weight matrices of the NNs.
        One extra row is added for bias. Individuals are then stacked together
        to form the population.

        Returns
        -------
        individuals : ndarray
            The population for EvoNN
        """
        individuals = np.random.uniform(
            self.w_low,
            self.w_high,
            size=(self.params["pop_size"], self.num_input_nodes, self.num_nodes),
        )

        # Randomly set some weights to zero
        zeros = np.random.choice(
            np.arange(individuals.size), ceil(individuals.size * self.params["prob_omit"])
        )
        individuals.ravel()[zeros] = 0

        # Set bias
        individuals = np.insert(individuals, 0, 1, axis=1)

        return individuals

    def train(self, model):
        """Trains the networks and selects the best model from the non dominated front.

        Parameters
        ----------
        model : :obj:
            The model to be chosen.
        """
        pop = Population(
            self, assign_type="EvoNN", pop_size=self.params["pop_size"], plotting=False
        )
        pop.evolve(
            PPGA,
            {
                "logging": self.params["logging"],
                "logfile": model.log,
                "iterations": 10,
                "generations_per_iteration": 10,
            },
        )

        non_dom_front = pop.non_dominated()
        model.w_matrix, model.fitness = self.select(
            pop, non_dom_front, self.params["criterion"]
        )

        non_linear_layer = self.activation(model.w_matrix)
        model.linear_layer, _, _ = self.minimize_error(non_linear_layer)

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

        non_linear_layer = self.activation(decision_variables)
        _, _, predicted_values = self.minimize_error(non_linear_layer)
        training_error = self.loss_function(predicted_values)
        complexity = self.calculate_complexity(decision_variables)

        obj_func = [training_error, complexity]

        return obj_func

    def activation(self, w_matrix):
        """ Calculates the dot product and applies the activation function.

        Parameters
        ----------
        w_matrix : ndarray
            Weight matrix of the neural network

        Returns
        -------
        The final non-linear layer before the output

        """
        # Calculate the dot product + bias
        out = np.dot(self.X_train, w_matrix[1:, :]) + w_matrix[0]

        return self.activate(self.params["activation_func"], out)

    def minimize_error(self, non_linear_layer):
        """ Minimize the training error.

        Parameters
        ----------
        non_linear_layer : ndarray
            Output of the activation function

        Returns
        -------
        w2[0] : ndarray
            The weight matrix of the upper part of the network
        rss : float
            Sum of residuals
        predicted_values : ndarray
            The prediction of the model
        """

        if self.params["opt_func"] == "llsq":
            linear_solution = np.linalg.lstsq(non_linear_layer, self.y_train, rcond=None)
            linear_layer = linear_solution[0]
            rss = linear_solution[1]
            predicted_values = np.dot(non_linear_layer, linear_layer)
            return linear_layer, rss, predicted_values

    def loss_function(self, predicted_values):
        """Calculate the error between prediction and target values."""

        if self.params["loss_func"] == "mse":
            return ((self.y_train - predicted_values) ** 2).mean()
        if self.params["loss_func"] == "rmse":
            return np.sqrt(((self.y_train - predicted_values) ** 2).mean())

    def calculate_complexity(self, w_matrix):
        """Calculate the complexity of the model.

        Returns
        -------
        The number of non-zero connections in the lower part of the network.
        """

        k = np.count_nonzero(w_matrix[1:, :])

        return k

    def information_criterion(self, decision_variables):
        """Calculate the information criterion.

        Currently supports Akaike and corrected Akaike Information Criterion.

        Returns
        -------
        Corrected Akaike Information Criterion by default
        """
        non_linear_layer = self.activation(decision_variables)
        linear_layer, rss, _ = self.minimize_error(non_linear_layer)
        # rss = ((self.y_train - prediction) ** 2).sum()
        k = self.calculate_complexity(decision_variables) + np.count_nonzero(linear_layer)
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
            Possible values: 'min_error', 'akaike_corrected'

        Returns
        -------
        The selected model
        """
        model = None
        fitness = None
        if criterion == "min_error":
            # Return the model with the lowest error

            lowest_error = np.argmin(pop.objectives[:, 0])
            model = pop.individuals[lowest_error]
            fitness = pop.fitness[lowest_error]

        elif criterion == "akaike_corrected":

            # Calculate Akaike information criterion for the non-dominated front
            # and return the model with the lowest value

            info_c_rank = []

            for i in non_dom_front:

                info_c = self.information_criterion(pop.individuals[i])
                info_c_rank.append((info_c, i))

            info_c_rank.sort()

            model = pop.individuals[info_c_rank[0][1]]
            fitness = pop.fitness[info_c_rank[0][1]]

        return model, fitness

    def create_logfile(self):
        """Create a log file containing the parameters for training the model and the GA.

        Returns
        -------
        An external log file
        """

        # Save params to log file
        log_file = open(
            self.name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.num_nodes)
            + ".log",
            "a",
        )
        print(
            "samples: "
            + str(self.num_of_samples)
            + "\n"
            + "variables: "
            + str(self.num_of_variables)
            + "\n"
            + "nodes: "
            + str(self.num_nodes)
            + "\n"
            + "activation: "
            + self.params["activation_func"]
            + "\n"
            + "opt func: "
            + self.params["opt_func"]
            + "\n"
            + "loss func: "
            + self.params["loss_func"],
            file=log_file,
        )
        return log_file

    @staticmethod
    def activate(name, x):
        if name == "sigmoid":
            return expit(x)

        if name == "relu":
            return np.maximum(x, 0)

        if name == "tanh":
            return np.tanh(x)


class EvoNNModel(EvoNN):
    """The class for the surrogate model.

    Parameters
    ----------
    name : str
        Name of the problem
    w_matrix : ndarray
        The weight matrix of the lower part of the network
    linear_layer : ndarray
        The linear layer of the upper part of the network

    """
    def __init__(self, name, w_matrix=None, linear_layer=None):
        super().__init__(name)
        self.name = name
        self.w_matrix = w_matrix
        self.linear_layer = linear_layer
        self.svr = None
        self.log = None
        self.set_params()

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.

        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values
        """
        prob = EvoNN(name=self.name, params=self.params)
        prob.fit(training_data, target_values)
        if prob.params["logging"]:
            self.log = prob.create_logfile()
        prob.train(self)

        self.single_variable_response(ploton=False, log=self.log)

    def predict(self, decision_variables):

        out = np.dot(decision_variables, self.w_matrix[1:, :]) + self.w_matrix[0]

        non_linear_layer = self.activate(self.params["activation_func"], out)

        y = np.dot(non_linear_layer, self.linear_layer)

        return y

    def set_params(
        self,
        name=None,
        pop_size=500,
        num_nodes=15,
        prob_omit=0.2,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="akaike_corrected",
        logging=False,
        plotting=False,
    ):

        """ Set parameters for EvoNN model.

        Parameters
        ----------
        name : str
            Name of the problem.
        pop_size : int
            Population size.
        num_nodes : int
            Maximum number of nodes per layer.
        prob_omit : float
            Probability of setting some weights to zero initially.
        activation_func : str
            Function to use for activation.
        opt_func : str
            Function to use for optimizing the final layer of the model.
        loss_func : str
            The loss function to use.
        criterion : str
            The criterion to use for selecting the model.
        logging : bool
            True to create a logfile, False otherwise.
        plotting : bool
            True to create a plot, False otherwise.
        """
        params = {
            "name": name,
            "pop_size": pop_size,
            "num_nodes": num_nodes,
            "prob_omit": prob_omit,
            "activation_func": activation_func,
            "opt_func": opt_func,
            "loss_func": loss_func,
            "criterion": criterion,
            "logging": logging,
            "plotting": plotting,
        }

        self.params = params

    def plot(self, prediction, target):
        """Creates and shows a plot for the model.

        Parameters
        ----------
        The model to create the plot for.
        """

        trace0 = go.Scatter(x=prediction, y=target, mode="markers")
        trace1 = go.Scatter(x=target, y=target)
        data = [trace0, trace1]
        plotly.offline.plot(
            data,
            filename=self.name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.num_nodes)
            + ".html",
            auto_open=True,
        )

    def single_variable_response(self, ploton=False, log=None):
        """Get the model's response to a single variable."""

        trend = np.loadtxt("trend")
        avg = np.ones((1, self.w_matrix[1:].shape[0])) * (0 + 1) / 2
        svr = np.empty((0, 2))

        for i in range(self.w_matrix[1:].shape[0]):
            variables = np.ones((len(trend), 1)) * avg
            variables[:, i] = trend

            out = self.predict(variables)

            if min(out) == max(out):
                out = 0.5 * np.ones(out.size)
            else:
                out = (out - min(out)) / (max(out) - min(out))

            if ploton:
                trace0 = go.Scatter(
                    x=np.arange(len(variables[:, 1])), y=variables[:, i], name="input"
                )
                trace1 = go.Scatter(
                    x=np.arange(len(variables[:, 1])), y=out, name="output"
                )
                data = [trace0, trace1]
                plotly.offline.plot(
                    data, filename="x" + str(i + 1) + "_response.html", auto_open=True
                )

            p = np.diff(out)
            q = np.diff(trend)
            r = np.multiply(p, q)
            r_max = max(r)
            r_min = min(r)
            response = None
            s = None
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
            svr = np.vstack((svr, ["x" + str(i + 1), s]))
            self.svr = svr
