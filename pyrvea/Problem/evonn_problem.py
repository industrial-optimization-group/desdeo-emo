from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from math import ceil
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
    prob_omit : float
        The probability of setting some weights to zero initially
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
        prob_omit=0.2,
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
        self.prob_omit = prob_omit
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
            np.arange(individuals.size), ceil(individuals.size * self.prob_omit)
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
        model.w1, model.fitness = self.select(
            pop, non_dom_front, self.params["criterion"]
        )

        activated_layer = self.activation(model.w1)
        model.w2, _, model.y_pred = self.minimize_error(activated_layer)

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

        Returns
        -------
        The penultimate layer before the output

        """
        w1 = decision_variables
        # Calculate the dot product + bias
        wi = np.dot(self.X_train, w1[1:, :]) + w1[0]

        if self.params["activation_func"] == "sigmoid":
            activated_layer = lambda x: 1 / (1 + np.exp(-x))

        if self.params["activation_func"] == "relu":
            activated_layer = lambda x: np.maximum(x, 0)

        if self.params["activation_func"] == "tanh":
            activated_layer = lambda x: np.tanh(x)

        return activated_layer(wi)

    def minimize_error(self, activated_layer):
        """ Minimize the training error.

        Parameters
        ----------
        activated_layer : ndarray
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
            w2 = np.linalg.lstsq(activated_layer, self.y_train, rcond=None)
            rss = w2[1]
            predicted_values = np.dot(activated_layer, w2[0])
            return w2[0], rss, predicted_values

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
        z = self.activation(decision_variables)
        w_matrix2, rss, prediction = self.minimize_error(z)
        # rss = ((self.y_train - prediction) ** 2).sum()
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

    def create_plot(self, model):
        """Creates and shows a plot for the model.

        Parameters
        ----------
        The model to create the plot for.
        """

        trace0 = go.Scatter(x=model.y_pred, y=self.y_train, mode="markers")
        trace1 = go.Scatter(x=self.y_train, y=self.y_train)
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


class EvoNNModel(EvoNN):
    """The class for the surrogate model.

    Parameters
    ----------
    name : str
        Name of the problem
    w1 : ndarray
        The weight matrix of the lower part of the network
    w2 : ndarray
        The weight matrix of the upper part of the network
    y_pred : ndarray
        Prediction of the model

    """
    def __init__(self, name, w1=None, w2=None, y_pred=None):
        super().__init__(name)
        self.name = name
        self.w1 = w1
        self.w2 = w2
        self.y_pred = y_pred
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
        if prob.params["plotting"]:
            prob.create_plot(self)
        self.num_of_samples = prob.num_of_samples
        self.single_variable_response(ploton=False, log=self.log)

    def predict(self, decision_variables):

        wi = np.dot(decision_variables, self.w1[1:, :]) + self.w1[0]

        if self.params["activation_func"] == "sigmoid":
            activated_layer = lambda x: 1 / (1 + np.exp(-x))

        if self.params["activation_func"] == "relu":
            activated_layer = lambda x: np.maximum(x, 0)

        if self.params["activation_func"] == "tanh":
            activated_layer = lambda x: np.tanh(x)

        out = np.dot(activated_layer(wi), self.w2)

        return out

    def set_params(
        self,
        name=None,
        pop_size=500,
        num_nodes=15,
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
            "activation_func": activation_func,
            "opt_func": opt_func,
            "loss_func": loss_func,
            "criterion": criterion,
            "logging": logging,
            "plotting": plotting,
        }

        self.params = params

    def single_variable_response(self, ploton=False, log=None):
        """Get the model's response to a single variable."""

        trend = np.loadtxt("trend")
        trend = trend[0 : self.num_of_samples]
        avg = np.ones((1, self.w1[1:].shape[0])) * (0 + 1) / 2
        svr = np.empty((0, 2))

        for i in range(self.w1[1:].shape[0]):
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
