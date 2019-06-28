from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from scipy.special import expit
import numpy as np
import plotly
import plotly.graph_objs as go
import random
from math import ceil


class EvoDN2(baseProblem):
    """Creates an Artificial Neural Network (ANN) for the EvoDN2 algorithm.
    Parameters
    ----------
    name : str
        Name of the sample
    num_of_objectives : int
        The number of objectives
    w_low : float
        The lower bound for randomly generated weights
    w_high : float
        The upper bound for randomly generated weights
    params :
    """

    def __init__(
        self,
        name,
        X_train=None,
        y_train=None,
        num_of_objectives=2,
        w_low=-5.0,
        w_high=5.0,
        params=None,
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
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
        self.subnet_struct = self.params["subnet_struct"]
        self.num_nodes = self.params["num_nodes"]

        # Create random subsets of decision variables for each subnet
        self.subsets = []
        for i in range(self.subnet_struct[0]):
            n = random.randint(1, self.X_train.shape[1])
            self.subsets.append(random.sample(range(self.X_train.shape[1]), n))

        # Ensure that each decision variable is used as an input in at least one subnet
        for n in list(range(self.X_train.shape[1])):
            if not any(n in k for k in self.subsets):
                self.subsets[random.randint(0, self.subnet_struct[0] - 1)].append(n)

        return self.subsets

    def train(self, model):

        pop = Population(
            self, assign_type="EvoDN2", pop_size=self.params["pop_size"], plotting=False
        )
        pop.evolve(
            PPGA,
            {
                "logging": self.params["logging"],
                "logfile": model.log,
                "iterations": 10,
                "generations_per_iteration": 10,
                "crossover_type": "short",
                "mutation_type": "short",
            },
        )

        non_dom_front = pop.non_dominated()
        model.subnets, model.fitness = self.select(
            pop, non_dom_front, self.params["criterion"]
        )
        model.non_linear_layer, _ = self.activation(model.subnets)
        model.linear_layer, _ = self.minimize_error(model.non_linear_layer)

    def create_population(self):

        individuals = []
        for i in range(self.params["pop_size"]):
            nets = []
            for j in range(self.subnet_struct[0]):

                layers = []
                num_layers = np.random.randint(1, self.subnet_struct[1])
                in_nodes = len(self.subsets[j])

                for k in range(num_layers):
                    out_nodes = random.randint(1, self.num_nodes)
                    net = np.random.uniform(
                        self.w_low, self.w_high, size=(in_nodes, out_nodes)
                    )
                    # Randomly set some weights to zero
                    zeros = np.random.choice(
                        np.arange(net.size), ceil(net.size * self.params["prob_omit"])
                    )
                    net.ravel()[zeros] = 0

                    # Add bias
                    net = np.insert(net, 0, 1, axis=0)
                    in_nodes = out_nodes
                    layers.append(net)
                nets.append(layers)

            individuals.append(nets)

        individuals = np.asarray(individuals)

        return individuals

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

        non_linear_layer, complexity = self.activation(decision_variables)
        _, predicted_values = self.minimize_error(non_linear_layer)
        training_error = self.loss_function(predicted_values)

        obj_func = [training_error, complexity]

        return obj_func

    def activation(self, decision_variables):
        """ Calculates the dot product and applies the activation function.
        Parameters
        ----------
        decision_variables : ndarray
            Array of all subnets in the current neural network

        Returns
        -------
        non_linear_layer : ndarray
            The final non-linear layer before the output
        complexity : float
            The complexity of the neural network
        """
        network_complexity = []
        non_linear_layer = np.empty((self.num_of_samples, 0))

        for i, subnet in enumerate(decision_variables):

            # Get the input variables for the first layer
            in_nodes = self.X_train[:, self.subsets[i]]
            subnet_complexity = 1

            for layer in subnet:
                # Calculate the dot product + bias
                out = np.dot(in_nodes, layer[1:, :]) + layer[0]

                subnet_complexity = np.dot(subnet_complexity, np.abs(layer[1:, :]))

                in_nodes = self.activate(self.params["activation_func"], out)

            network_complexity.append(np.sum(subnet_complexity))
            non_linear_layer = np.hstack((non_linear_layer, in_nodes))

        complexity = np.sum(network_complexity)

        return non_linear_layer, complexity

    def minimize_error(self, activated_layer):
        """ Minimize the training error.
        Parameters
        ----------
        activated_layer : ndarray
            Output of the activation function
        Returns
        -------
        w_matrix[0] : ndarray
            The weight matrix of the upper part of the network
        rss : float
            Sums of residuals
        predicted_values : ndarray
            The prediction of the model
        """

        if self.params["opt_func"] == "llsq":
            w2 = np.linalg.lstsq(activated_layer, self.y_train, rcond=None)
            predicted_values = np.dot(activated_layer, w2[0])
            return w2[0], predicted_values

    def loss_function(self, predicted_values):

        if self.params["loss_func"] == "mse":
            return ((self.y_train - predicted_values) ** 2).mean()
        if self.params["loss_func"] == "rmse":
            return np.sqrt(((self.y_train - predicted_values) ** 2).mean())

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
        fitness = None
        if criterion == "min_error":
            # Return the model with the lowest error

            lowest_error = np.argmin(pop.objectives[:, 0])
            model = pop.individuals[lowest_error]
            fitness = pop.fitness[lowest_error]

        return model, fitness

    def create_logfile(self):

        # Save params to log file
        log_file = open(
            self.name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.subnet_struct[0])
            + "_"
            + str(self.subnet_struct[1])
            + "_"
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
            + "number of subnets: "
            + str(self.subnet_struct[0])
            + "\n"
            + "max number of layers: "
            + str(self.subnet_struct[1])
            + "\n"
            + "max nodes: "
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


class EvoDN2Model(EvoDN2):
    """Class for the surrogate model.
    Parameters
    ----------
    name : str
        Name of the problem
    subnets : ndarray
        The subnets of the model
    linear_layer : ndarray
        The final optimized layer of the network
    y_pred : ndarray
        Prediction of the model
    """

    def __init__(self, name, subnets=None, linear_layer=None, y_pred=None):
        super().__init__(name)
        self.name = name
        self.subnets = subnets
        self.subsets = None
        self.fitness = None
        self.non_linear_layer = None
        self.linear_layer = linear_layer
        self.y_pred = y_pred
        self.svr = None
        self.log = None
        self.set_params()

    def set_params(
        self,
        name=None,
        pop_size=500,
        subnet_struct=(4, 8),
        num_nodes=10,
        prob_omit=0.2,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="min_error",
        logging=False,
        plotting=False,
    ):
        """ Set parameters for EvoDN2 model.
        Parameters
        ----------
        name : str
            Name of the problem.
        pop_size : int
            Population size.
        subnet_struct : tuple
            Structure of the subnets for the model, shape=(num of subnets, max num of layers)
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
            "subnet_struct": subnet_struct,
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

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.
        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values
        """
        prob = EvoDN2(name=self.name, params=self.params)
        self.subsets = prob.fit(training_data, target_values)
        self.num_of_variables = prob.num_of_variables
        if self.params["logging"]:
            self.log = prob.create_logfile()

        prob.train(self)

        self.single_variable_response(ploton=False, log=self.log)

    def plot(self, prediction, target):

        trace0 = go.Scatter(x=prediction, y=target, mode="markers")
        trace1 = go.Scatter(x=target, y=target)
        data = [trace0, trace1]
        plotly.offline.plot(
            data,
            filename=self.name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.params["subnet_struct"][0])
            + "_"
            + str(self.params["subnet_struct"][1])
            + "_"
            + str(self.params["num_nodes"])
            + ".html",
            auto_open=True,
        )

    def predict(self, decision_variables):

        non_linear_layer = np.empty((decision_variables.shape[0], 0))

        for i, subnet in enumerate(self.subnets):

            in_nodes = decision_variables[:, self.subsets[i]]

            for layer in subnet:

                out = np.dot(in_nodes, layer[1:, :]) + layer[0]

                in_nodes = self.activate(self.params["activation_func"], out)

            non_linear_layer = np.hstack((non_linear_layer, in_nodes))

        y = np.dot(non_linear_layer, self.linear_layer)

        return y

    def single_variable_response(self, ploton=False, log=None):

        trend = np.loadtxt("trend")
        avg = np.ones((1, self.num_of_variables)) * (np.finfo(float).eps + 1) / 2
        svr = np.empty((0, 2))

        for i in range(self.num_of_variables):
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
