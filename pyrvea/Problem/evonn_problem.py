from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from scipy.special import expit
import numpy as np
import plotly
import plotly.graph_objs as go


class EvoNN(BaseProblem):
    """Creates Artificial Neural Network (ANN) models for the EvoNN algorithm.

    These models contain only one hidden node layer. The lower part of the network
    is optimized by a genetic algorithm, and the upper part is optimized by Linear Least Square
    algorithm by default.

    Parameters
    ----------
    name : str
        Name of the problem.
    X_train : ndarray
        Training data input.
    y_train : ndarray
        Training data target values.
    params : dict
        Parameters for model training.
    num_samples : int
        The number of data points, or samples.
    num_of_objectives : int
        The number of objectives.

    References
    ----------
    [1] N. Chakraborti. Strategies for Evolutionary Data Driven Modeling in Chemical and Metallurgical Systems.
    J. Valadi and P. Siarry (eds.), Applications of Metaheuristics in Process Engineering, pp. 89-122, 2014.
    [2] F. Pettersson, N. Chakraborti, H. Saxén. A genetic algorithms based multi-objective neural
    net applied to noisy blast furnace data. Applied Soft Computing 7, pp. 387–397, 2007.

    """

    def __init__(
        self,
        name=None,
        X_train=None,
        y_train=None,
        num_of_objectives=2,
        params=None,
        num_samples=None,
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_objectives = num_of_objectives
        self.params = params
        self.num_samples = num_samples

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
        _, _, predicted_values, training_error = self.optimize_layer(non_linear_layer)

        obj_func = [training_error, complexity]

        return obj_func

    def activation(self, non_linear_layer):
        """ Calculates the dot product and applies the activation function.
        Also get complexity for the lower part of the network.

        Parameters
        ----------
        non_linear_layer : ndarray
            Weight matrix of the neural network

        Returns
        -------
        activated_layer : ndarray
            The activated non-linear layer before the output.
        complexity : int
            The model's complexity

        """
        # Calculate the dot product + bias
        out = np.dot(self.X_train, non_linear_layer[1:, :]) + non_linear_layer[0]
        complexity = np.count_nonzero(non_linear_layer[1:, :])
        activated_layer = self.activate(self.params["activation_func"], out)

        return activated_layer, complexity

    def optimize_layer(self, non_linear_layer):
        """ Apply the linear function to the activated layer
        and calculate the training error.

        Parameters
        ----------
        non_linear_layer : ndarray
            Output of the activation function

        Returns
        -------
        linear_layer : ndarray
            The optimized weight matrix of the upper part of the network
        rss : float
            Sum of residuals
        predicted_values : ndarray
            The prediction of the model
        training_error : float
            The model's training error
        """

        linear_layer = None
        rss = None
        predicted_values = None
        training_error = None

        if self.params["opt_func"] == "llsq":
            linear_solution = np.linalg.lstsq(
                non_linear_layer, self.y_train, rcond=None
            )
            linear_layer = linear_solution[0]
            rss = linear_solution[1]
            predicted_values = np.dot(non_linear_layer, linear_layer)

        if self.params["loss_func"] == "rmse":
            training_error = np.sqrt(((self.y_train - predicted_values) ** 2).mean())

        return linear_layer, rss, predicted_values, training_error

    def information_criterion(self, decision_variables):
        """Calculate the information criterion.

        Currently supports Akaike and corrected Akaike Information Criterion.

        Returns
        -------
        Corrected Akaike Information criterion
        """
        non_linear_layer, complexity = self.activation(decision_variables)
        linear_layer, rss, _, _ = self.optimize_layer(non_linear_layer)
        k = complexity + np.count_nonzero(linear_layer)
        aic = 2 * k + self.num_samples * np.log(rss / self.num_samples)
        aicc = aic + (2 * k * (k + 1) / (self.num_samples - k - 1))

        return aicc

    def select(self, pop, non_dom_front, selection="min_error"):
        """ Select target model from the population.

        Parameters
        ----------
        pop : obj
            The population object.
        non_dom_front : list
            Indices of the models on the non-dominated front.
        selection : str
            The selection to use for selecting the model.
            Possible values: 'min_error', 'akaike_corrected', 'manual'

        Returns
        -------
        The selected model
        """
        model = None
        fitness = None
        if selection == "min_error":
            # Return the model with the lowest error

            lowest_error = np.argmin(pop.objectives[:, 0])
            model = pop.individuals[lowest_error]
            fitness = pop.fitness[lowest_error]

        elif selection == "akaike_corrected":

            # Calculate Akaike information criterion for the non-dominated front
            # and return the model with the lowest value

            info_c_rank = []

            for i in non_dom_front:

                info_c = self.information_criterion(pop.individuals[i])
                info_c_rank.append((info_c, i))

            info_c_rank.sort()

            model = pop.individuals[info_c_rank[0][1]]
            fitness = pop.fitness[info_c_rank[0][1]]

        elif selection == "manual":

            pareto = pop.objectives[non_dom_front]
            hover = pop.objectives.tolist()

            for i, x in enumerate(hover):
                x.insert(0, "Model " + str(i) + "<br>")

            trace0 = go.Scatter(
                x=pop.objectives[:, 0], y=pop.objectives[:, 1], mode="markers"
            )
            trace1 = go.Scatter(
                x=pareto[:, 0],
                y=pareto[:, 1],
                text=hover,
                hoverinfo="text",
                mode="markers+lines",
            )
            data = [trace0, trace1]
            layout = go.Layout(
                xaxis=dict(title="training error"), yaxis=dict(title="complexity")
            )
            plotly.offline.plot(
                {"data": data, "layout": layout},
                filename=self.name + "_training_models_" + "pareto" + ".html",
                auto_open=True,
            )

            model_idx = None
            while model_idx not in pop.objectives:
                usr_input = input(
                    "Please input the number of the model of your preference: "
                )
                try:
                    model_idx = int(usr_input)
                except ValueError:
                    print("Invalid input, please enter the model number.")
                    continue

                if model_idx not in pop.objectives:
                    print("Model " + str(model_idx) + " not found.")

            model = pop.individuals[int(model_idx)]
            fitness = pop.fitness[int(model_idx)]

        return model, fitness

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
    **kwargs
        Parameters passed for the model.

    Attributes
    ----------
    name : str
        Name of the model.
    non_linear_layer : ndarray
        The weight matrix of the lower part of the network.
    linear_layer : ndarray
        The linear layer of the upper part of the network.
    fitness : list
        Fitness of the trained model.
    svr : ndarray
        Single variable response of the model.
    log : file
        If logging set to True in params, external log file is stored here.

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.name = "EvoNN_Model"
        self.non_linear_layer = None
        self.linear_layer = None
        self.fitness = None
        self.svr = None
        self.log = None
        self.set_params(**kwargs)

    def set_params(
        self,
        name="EvoNN_Model",
        algorithm=PPGA,
        pop_size=500,
        num_nodes=20,
        prob_omit=0.3,
        w_low=-5.0,
        w_high=5.0,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        selection="akaike_corrected",
        recombination_type="evonn_nodeswap_gaussian",
        crossover_type="evonn_xover_nodeswap",
        mutation_type="evonn_mut_gaussian",
        logging=False,
        plotting=False,
        ea_parameters=None
    ):

        """ Set parameters for the EvoNN model.

        Parameters
        ----------
        name : str
            Name of the problem.
        algorithm : :obj:
            Which evolutionary algorithm to use for training the models.
        pop_size : int
            Population size.
        num_nodes : int
            Maximum number of nodes per layer.
        prob_omit : float
            Probability of setting some weights to zero initially.
        w_low : float
            The lower bound for randomly generated weights.
        w_high : float
            The upper bound for randomly generated weights.
        activation_func : str
            Function to use for activation.
        opt_func : str
            Function to use for optimizing the final layer of the model.
        loss_func : str
            The loss function to use.
        selection : str
            The selection to use for selecting the model.
        recombination_type, crossover_type, mutation_type : str or None
            Recombination functions. If recombination_type is specified, crossover and mutation
            will be handled by the same function. If None, they are done separately.
        logging : bool
            True to create a logfile, False otherwise.
        plotting : bool
            True to create a plot, False otherwise.
        ea_parameters : dict
            Contains the parameters needed by EA (Default value = None).
        """

        params = {
            "name": name,
            "algorithm": algorithm,
            "pop_size": pop_size,
            "num_nodes": num_nodes,
            "prob_omit": prob_omit,
            "w_low": w_low,
            "w_high": w_high,
            "activation_func": activation_func,
            "opt_func": opt_func,
            "loss_func": loss_func,
            "selection": selection,
            "recombination_type": recombination_type,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "logging": logging,
            "plotting": plotting,
            "ea_parameters": ea_parameters
        }

        self.name = name
        self.params = params

    def fit(self, training_data, target_values):
        """Fit data in EvoNN model.

        Parameters
        ----------
        training_data : ndarray, shape = (numbers of samples, number of variables)
            Training data
        target_values : ndarray
            Target values

        Returns
        -------
        self : returns an instance of self.

        """

        self.X_train = training_data
        self.y_train = target_values
        self.num_samples = target_values.shape[0]
        self.num_of_variables = training_data.shape[1]

        if self.params["logging"]:
            self.log = self.create_logfile()

        self.train()

        if self.params["logging"]:
            print(self.fitness, file=self.log)

        return self

    def train(self):
        """Trains the networks and selects the best model from the non dominated front.

        """
        pop = Population(
            self,
            assign_type="EvoNN",
            pop_size=self.params["pop_size"],
            plotting=self.params["plotting"],
            recombination_type=self.params["recombination_type"],
            crossover_type=self.params["crossover_type"],
            mutation_type=self.params["mutation_type"],
        )
        pop.evolve(
            EA=self.params["algorithm"],
            ea_parameters=self.params["ea_parameters"]

        )

        non_dom_front = pop.non_dominated()
        self.non_linear_layer, self.fitness = self.select(
            pop, non_dom_front, self.params["selection"]
        )

        activated_layer, _ = self.activation(self.non_linear_layer)
        self.linear_layer, *_ = self.optimize_layer(activated_layer)

    def predict(self, decision_variables):
        """Predict using the EvoNN model.

        Parameters
        ----------
        decision_variables : ndarray
            The decision variables used for prediction.

        Returns
        -------
        y : ndarray
            The prediction of the model.

        """
        out = (
            np.dot(decision_variables, self.non_linear_layer[1:, :])
            + self.non_linear_layer[0]
        )

        non_linear_layer = self.activate(self.params["activation_func"], out)  # rename

        y = np.dot(non_linear_layer, self.linear_layer)

        return y

    def plot(self, prediction, target, name=None):
        """Creates and shows a plot for the model's prediction.

        Parameters
        ----------
        prediction : ndarray
            The prediction of the model.
        target : ndarray
            The target values.
        name : str
            Filename to save the plot as.
        """

        if name is None:
            name = self.name

        trace0 = go.Scatter(x=prediction, y=target, mode="markers")
        trace1 = go.Scatter(x=target, y=target)
        data = [trace0, trace1]
        plotly.offline.plot(
            data,
            filename="Tests/"
            + self.params["algorithm"].__name__
            + self.__class__.__name__
            + name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.params["num_nodes"])
            + ".html",
            auto_open=True,
        )

    def create_logfile(self, name=None):
        """Create a log file containing the parameters for training the model and the EA.

        Parameters
        ----------
        name : str
            Filename to save the log as.

        Returns
        -------
        log_file : file
            An external log file.
        """

        if name is None:
            name = self.name

        # Save params to log file
        log_file = open(
            "Tests/"
            + self.params["algorithm"].__name__
            + self.__class__.__name__
            + name
            + "_var"
            + str(self.num_of_variables)
            + "_nodes"
            + str(self.params["num_nodes"])
            + ".log",
            "a",
        )

        print(
            "samples: "
            + str(self.num_samples)
            + "\n"
            + "variables: "
            + str(self.num_of_variables)
            + "\n"
            + "nodes: "
            + str(self.params["num_nodes"])
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

    def single_variable_response(self, ploton=False, log=None):
        """Get the model's response to a single variable.

        Parameters
        ----------
        ploton : bool
            Create and show plot on/off.
        log : file
            Write the results in a log file.

        """

        trend = np.loadtxt("trend")
        avg = np.ones((1, self.non_linear_layer[1:].shape[0])) * (0 + 1) / 2
        svr = np.empty((0, 2))

        for i in range(self.non_linear_layer[1:].shape[0]):
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

            if log is not None:
                print(
                    "x" + str(i + 1) + " response: " + str(response) + " " + s, file=log
                )
            svr = np.vstack((svr, ["x" + str(i + 1), s]))
            self.svr = svr
