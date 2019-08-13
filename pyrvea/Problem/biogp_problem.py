from random import random, randint, choice, seed
from graphviz import Digraph, Source
import numpy as np
import pandas as pd
from math import ceil
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.TournamentEA import TournamentEA
from pyrvea.Population.Population import Population
import plotly
import plotly.graph_objs as go


class BioGP(BaseProblem):
    """Creates syntax tree models to use for genetic programming through bi-objective genetic algorithms.

    The BioGP technique initially minimizes training error through a single objective optimization procedure and then a
    trade-off between complexity and accuracy is worked out through a genetic algorithm based bi-objective
    optimization strategy.

    The benefit of the BioGP approach is that an expert user or a decision maker (DM) can
    flexibly select the mathematical operations involved to construct a meta-model of desired complexity or
    accuracy. It is also designed to combat bloat – a perennial problem in genetic programming along with
    over fitting and under fitting problems.

    Notes
    -----
    The algorithm has been created earlier in MATLAB, and this Python implementation has been using
    that code as a basis.

    Python code has been written by Niko Rissanen under the supervision of professor Nirupam Chakraborti.

    Parameters
    ----------
    name : str
        Name of the problem.
    X_train : np.ndarray
        Training data input.
    y_train : np.ndarray
        Training data target values.
    num_of_objectives : int
        The number of objectives.
    params : dict
        Parameters for model training.
    num_samples : int
        The number of data points, or samples.
    function_set : array_like
        The function set to use when creating the trees.
    terminal_set : array_like
        The terminals (variables and constants) to use when creating the trees.

    References
    ----------
    [1] Giri, B. K., Hakanen, J., Miettinen, K., & Chakraborti, N. (2013). Genetic programming through bi-objective
    genetic algorithms with a study of a simulated moving bed process involving multiple objectives.
    Applied Soft Computing, 13(5), 2613-2623.
    """

    def __init__(
        self,
        name=None,
        X_train=None,
        y_train=None,
        num_of_objectives=2,
        params=None,
        num_samples=None,
        function_set=None,
        terminal_set=None,
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_objectives = num_of_objectives
        self.params = params
        self.num_samples = num_samples
        # These functions should be moved to their own separate package maybe
        self.function_map = {
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "sqrt": self.sqrt,
            "log": self.log,
            "sin": self.sin,
            "cos": self.cos,
            "tan": self.tan,
            "neg": self.neg,
        }
        self.terminal_set = terminal_set
        self.function_set = function_set

        self.individuals = []

    def create_individuals(self):

        if self.params["init_method"] == "ramped_half_and_half":
            for md in range(
                ceil(self.params["max_depth"] / 2), self.params["max_depth"] + 1
            ):
                for i in range(
                    int(self.params["pop_size"] / (self.params["max_depth"] + 1))
                ):

                    ind = LinearNode(value="linear", params=self.params)
                    ind.grow_tree(max_depth=md, method="grow", ind=ind)
                    self.individuals.append(ind)

                for i in range(
                    int(self.params["pop_size"] / (self.params["max_depth"] + 1))
                ):

                    ind = LinearNode(value="linear", params=self.params)
                    ind.grow_tree(max_depth=md, method="full", ind=ind)
                    self.individuals.append(ind)

        elif self.params["init_method"] == "full":
            for i in range(self.params["pop_size"]):
                ind = LinearNode(value="linear", params=self.params)
                ind.grow_tree(method="full", ind=ind)
                self.individuals.append(ind)

        elif self.params["init_method"] == "grow":
            for i in range(self.params["pop_size"]):
                ind = LinearNode(value="linear", params=self.params)
                ind.grow_tree(method="grow", ind=ind)
                self.individuals.append(ind)

        return self.individuals

    def objectives(self, decision_variables):

        return decision_variables.calculate_linear(self.X_train, self.y_train)

    def select(self, pop, non_dom_front, selection="min_error"):
        """ Select target model from the population.

        Parameters
        ----------
        pop : obj
            The population object.
        non_dom_front : list
            Indices of the models on the non-dominated front.
        selection : str
            The criterion to use for selecting the model.
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

        return model, fitness

    @staticmethod
    def add(x, y):
        return np.add(x, y)

    @staticmethod
    def sub(x, y):
        return np.subtract(x, y)

    @staticmethod
    def mul(x, y):
        return np.multiply(x, y)

    @staticmethod
    def div(x, y):
        y[y == 0] = 1.0
        return np.divide(x, y)

    @staticmethod
    def sqrt(x):
        return np.sqrt(np.abs(x))

    @staticmethod
    def log(x):
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)

    @staticmethod
    def sin(x):
        return np.sin(x)

    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def tan(x):
        return np.tan(x)

    @staticmethod
    def neg(x):
        return np.negative(x)


class BioGPModel(BioGP):
    """The class for the BioGP surrogate model.

    Parameters
    ----------
    model_parameters : dict
        Parameters passed for the model.
    ea_parameters : dict
        Parameters passed for the genetic algorithm.

    Attributes
    ----------
    name : str
        Name of the model.
    linear_node : LinearNode
        The parent node of the tree containing the linear values.
    fitness : list
        Fitness of the trained model.
    minimize : list (bool)
        List of which objectives to minimize.
    svr : np.ndarray
        Single variable response of the model.
    log : file
        If logging set to True in params, external log file is stored here.

    """
    def __init__(self, model_parameters=None, ea_parameters=None):
        super().__init__()
        self.name = "BioGP_Model"
        self.linear_node = None
        self.fitness = None
        self.minimize = None
        self.svr = None
        self.log = None
        self.ea_params = ea_parameters
        if model_parameters:
            self.set_params(**model_parameters)
        else:
            self.set_params()

    def set_params(
        self,
        name="BioGP_Model",
        training_algorithm=PPGA,
        pop_size=500,
        max_depth=5,
        max_subtrees=4,
        prob_terminal=0.5,
        complexity_scalar=0.5,
        error_lim=0.001,
        init_method="ramped_half_and_half",
        selection="min_error",
        loss_func="root_mean_square",
        recombination_type=None,
        crossover_type="biogp_xover",
        mutation_type="biogp_mut",
        single_obj_generations=10,
        logging=False,
        plotting=False,
        function_set=("add", "sub", "mul", "div"),
        terminal_set=None,
    ):

        """ Set parameters for the BioGP model.

        Parameters
        ----------
        name : str
            Name of the problem.
        training_algorithm : EA
            Which evolutionary algorithm to use for training the models.
        pop_size : int
            Population size.
        max_depth : int
            Maximum depth of the tree.
        max_subtrees : int
            Maximum number of subtrees the tree can grow.
        prob_terminal : float
            The probability of making the node terminal when growing the tree.
        complexity_scalar : float
            Complexity of the model is calculated as a weighted aggregate of the maximum depth of the GP tree
            and the total number of corresponding function nodes. Larger value gives more weight to the depth,
            lower more weight to the number of function nodes.
        error_lim : float
            Used to control bloat. If the error reduction ratio of a subtree is less than this value,
            then that root is terminated and a new root is grown under the linear node (i.e., parent node).
        init_method : str
            Method to use for creating the initial population. Can be either 'grow', 'full', or
            'ramped_half_and_half' (default).

            grow: nodes are chosen at random from both functions and terminals, allowing for smaller trees
            than max_depth.
            full: nodes are chosen from the function set until the max depth is reached, and then terminals are chosen.
            ramped half and half: trees are grown with a 50/50 mixture of 'grow' and 'full'.

        loss_func : str
            The loss function to use.
        selection : str
            The selection method to use.
        recombination_type, crossover_type, mutation_type : str or None
            Recombination functions. If recombination_type is specified, crossover and mutation
            will be handled by the same function. If None, they are done separately.
        single_obj_generations : int
            How many generations to run minimizing only the training error.
        logging : bool
            True to create a logfile, False otherwise.
        plotting : bool
            True to create a plot, False otherwise.
        function_set : tuple
            The function set to use when creating the trees.
        terminal_set : list
            The terminals (variables and constants) to use when creating the trees.

        """

        params = {
            "name": name,
            "training_algorithm": training_algorithm,
            "pop_size": pop_size,
            "max_depth": max_depth,
            "max_subtrees": max_subtrees,
            "prob_terminal": prob_terminal,
            "complexity_scalar": complexity_scalar,
            "error_lim": error_lim,
            "init_method": init_method,
            "loss_func": loss_func,
            "selection": selection,
            "recombination_type": recombination_type,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "single_obj_generations": single_obj_generations,
            "logging": logging,
            "plotting": plotting,
            "function_set": function_set,
            "terminal_set": terminal_set,
        }

        self.params = params

    def fit(self, training_data, target_values):
        """Fit data in BioGP model.

        Parameters
        ----------
        training_data : pd.DataFrame, shape = (numbers of samples, number of variables)
            Training data.
        target_values : pd.DataFrame
            Target values.

        Returns
        -------
        self : returns an instance of self.

        """

        self.X_train = training_data
        self.y_train = target_values
        self.num_samples = target_values.shape[0]
        self.num_of_variables = training_data.shape[1]
        terminal_set = self.X_train.columns.tolist()
        if self.params["terminal_set"]:
            self.params["terminal_set"].extend(terminal_set)
        function_set = []
        for function in self.params["function_set"]:
            function_set.append(self.function_map[function])
        self.params["function_set"] = function_set

        self.train()

        self.single_variable_response(ploton=self.params["plotting"])

        if self.params["logging"]:
            self.create_logfile()

        return self

    def train(self):
        """Trains the networks and selects the best model from the non dominated front.

        """

        # Minimize only error for first n generations before switching to bi-objective
        ea_params = {
            "generations_per_iteration": 1,
            "iterations": self.params["single_obj_generations"],
        }
        self.minimize = [True, False]

        print(
            "Minimizing error for "
            + str(self.params["single_obj_generations"])
            + " generations..."
        )

        pop = Population(
            self,
            assign_type="BioGP",
            pop_size=self.params["pop_size"],
            plotting=self.params["plotting"],
            recombination_type=self.params["recombination_type"],
            crossover_type=self.params["crossover_type"],
            mutation_type=self.params["mutation_type"],
        )

        pop.evolve(EA=TournamentEA, ea_parameters=ea_params)

        # Switch to bi-objective (error, complexity)
        self.minimize = [True, True]
        pop.update_fitness()

        print("Switching to bi-objective mode")

        pop.evolve(
            EA=self.params["training_algorithm"], ea_parameters=self.ea_params
        )

        non_dom_front = pop.non_dominated()
        self.linear_node, self.fitness = self.select(
            pop, non_dom_front, self.params["selection"]
        )

    def predict(self, decision_variables):
        """Predict using the BioGP model.

        Parameters
        ----------
        decision_variables : pd.DataFrame
            The decision variables used for prediction.

        Returns
        -------
        y : np.ndarray
            The prediction of the model.

        """

        sub_trees = []
        # Stack outputs of subtrees to form weight matrix
        for root in self.linear_node.roots:
            sub_tree = root.predict(decision_variables)
            if sub_tree.size == 1:
                sub_tree = sub_tree.reshape(1)
            sub_trees.append(sub_tree)

        sub_trees = np.hstack(sub_trees)
        axis = None
        if sub_trees.ndim > 1:
            axis = 1
        sub_trees = np.insert(sub_trees, 0, 1, axis=axis)

        y = np.dot(sub_trees, self.linear_node.linear)

        if isinstance(y, float):
            y = np.asarray([y])

        return y

    def plot(self, prediction, target, name=None):
        """Creates and shows a plot for the model's prediction.

        Parameters
        ----------
        prediction : np.ndarray
            The prediction of the model.
        target : pd.DataFrame
            The target values.
        name : str
            Filename to save the plot as.
        """
        target = np.asarray(target)
        if name is None:
            name = self.name

        trace0 = go.Scatter(x=prediction, y=target, mode="markers")
        trace1 = go.Scatter(x=target, y=target)
        data = [trace0, trace1]
        plotly.offline.plot(
            data,
            filename="Tests/"
            + self.params["training_algorithm"].__name__
            + self.__class__.__name__
            + name
            + "_var"
            + str(self.num_of_variables)
            + "_depth"
            + str(self.params["max_depth"])
            + "_subtrees"
            + str(self.params["max_subtrees"])
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
            + self.params["training_algorithm"].__name__
            + self.__class__.__name__
            + name
            + "_var"
            + str(self.num_of_variables)
            + "_depth"
            + str(self.params["max_depth"])
            + "_subtrees"
            + str(self.params["max_subtrees"])
            + ".log",
            "a",
        )

        for i in self.params:
            print("", i, ":", self.params[i], file=log_file)

        if self.fitness is not None:
            print("fitness: " + str(self.fitness), file=log_file)

        if self.svr is not None:
            print("single variable response: " + str(self.svr), file=log_file)

        return log_file

    def single_variable_response(self, ploton=False):
        """Get the model's response to a single variable.

        Parameters
        ----------
        ploton : bool
            Create and show plot on/off.
        """

        trend = np.loadtxt("trend")
        avg = np.ones((1, self.num_of_variables)) * (np.finfo(float).eps + 1) / 2
        svr = np.empty((0, 2))
        variables = np.ones((len(trend), 1)) * avg
        dataset = pd.DataFrame.from_records(variables)
        dataset.columns = self.X_train.columns

        for i in range(self.num_of_variables):

            dataset.iloc[:, i] = trend

            out = self.predict(dataset)

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
            s = None
            if r_max <= 0 and r_min <= 0:
                s = "inverse"
            elif r_max >= 0 and r_min >= 0:
                s = "direct"
            elif r_max == 0 and r_min == 0:
                s = "nil"
            elif r_min < 0 < r_max:
                s = "mixed"

            svr = np.vstack((svr, ["x" + str(i + 1), s]))
            self.svr = svr


class Node:
    """A node object representing a function or terminal node in the tree.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables which are either float or str.
    depth : int
        The depth the node is at.
    params : None or dict
        The parameters of the model.
    function_set : array_like
        The function set to use when creating the trees.
    terminal_set : array_like
        The terminals (variables and constants) to use when creating the trees.

    """

    def __init__(
        self, value=None, depth=None, params=None, function_set=None, terminal_set=None
    ):
        self.value = value
        self.depth = depth
        self.params = params
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.nodes = []
        self.nodes_at_depth = None
        self.total_depth = None
        self.roots = []

    def predict(self, decision_variables=None):

        if callable(self.value):
            values = [root.predict(decision_variables) for root in self.roots]
            if self.depth == 0:
                self.value = sum(values)
                return self.value
            else:
                return self.value(*values)

        else:
            if isinstance(decision_variables, np.ndarray) and isinstance(
                self.value, str
            ):
                return decision_variables[
                    :, int("".join(filter(str.isdigit, self.value))) - 1
                ]
            if isinstance(self.value, str):
                return np.asarray(decision_variables[self.value]).reshape(-1, 1)
            else:
                return np.full((decision_variables.shape[0], 1), self.value)

    def node_label(self):  # return string label
        if callable(self.value):
            return self.value.__name__
        else:
            return str(self.value)

    def draw(self, dot, count):  # dot & count are lists in order to pass "by reference"
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())

        for node in self.roots:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            node.draw(dot, count)

    def draw_tree(self, name="tree", footer=""):
        dot = [Digraph()]
        dot[0].attr(kw="graph", label=footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename=name + ".gv", format="png").render()

    def get_sub_nodes(self):
        """Get all nodes belonging to the subtree under the current node.

        Returns
        -------
        nodes : list
            A list of nodes in the subtree.

        """
        nodes = []
        nodes_at_depth = {}
        stack = [self]
        while stack:
            cur_node = stack[0]
            depth = cur_node.depth
            if cur_node.depth not in nodes_at_depth:
                nodes_at_depth[cur_node.depth] = []
            nodes_at_depth[cur_node.depth].append(cur_node)
            stack = stack[1:]
            nodes.append(cur_node)
            if len(cur_node.roots) > 0:
                depth += 1
            for child in cur_node.roots:
                child.depth = depth
                stack.append(child)

        self.nodes_at_depth = nodes_at_depth
        self.total_depth = max(key for key in self.nodes_at_depth.keys())

        return nodes

    def grow_tree(self, max_depth=None, method="grow", depth=0, ind=None):
        """Create a random tree recursively using either grow or full method.

        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree.
        method : str
            Methods: 'grow', 'full'.
            For the 'grow' method, nodes are chosen at random from both functions and terminals.
            The 'full' method chooses nodes from the function set until the max depth is reached,
            and then terminals are chosen.
        depth : int
            Current depth.
        ind : :obj:
            The starting node from which to begin growing trees.

        """
        node = None
        if max_depth is None:
            max_depth = self.params["max_depth"]
        if depth == 0:
            if ind is None:
                ind = LinearNode(value="linear")
            num_subtrees = self.params["max_subtrees"]
            for i in range(len(ind.roots), num_subtrees):
                node = self.grow_tree(max_depth, method, depth=depth + 1)
                ind.roots.append(node)

        # Make terminal node
        elif (
            depth >= max_depth
            or method == "grow"
            and random() < self.params["prob_terminal"]
        ):
            node = Node(
                depth=depth,
                function_set=self.params["function_set"],
                terminal_set=self.params["terminal_set"],
            )
            node.value = choice(node.terminal_set)

        # Make function node
        else:
            node = Node(
                depth=depth,
                function_set=self.params["function_set"],
                terminal_set=self.params["terminal_set"],
            )
            node.value = choice(node.function_set)

            for i in range(node.value.__code__.co_argcount):  # Check arity
                root = self.grow_tree(max_depth, method, depth=depth + 1)
                node.roots.append(root)

        return node


class LinearNode(Node):
    """The parent node of the tree, from which a number of subtrees emerge, as defined
    by the user. The linear node takes a weighted sum of the output from the subtrees and
    also uses a bias value. The weights and the bias are calculated by the linear least
    square technique.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables which are either float or str.
    depth : int
        The depth the node is at.
    params : None or dict
        Parameters of the model.
    """

    def __init__(self, value="linear", depth=0, params=None):
        super().__init__(params=params)
        self.value = value
        self.nodes_at_depth = {}
        self.params = params
        self.depth = depth
        self.total_depth = None
        self.out = None
        self.complexity = None
        self.fitness = None
        self.linear = None

    def calculate_linear(self, X_train, y_train):

        sub_trees = []
        # Stack outputs of subtrees to form weight matrix
        for root in self.roots:
            sub_trees.append(root.predict(X_train))

        sub_trees = np.hstack(sub_trees)
        axis = None
        if sub_trees.ndim > 1:
            axis = 1
        sub_trees = np.insert(sub_trees, 0, 1, axis=axis)

        weights, *_ = np.linalg.lstsq(sub_trees, y_train, rcond=None)
        self.linear = weights
        out = np.dot(sub_trees, weights)

        # Error reduction ratio
        q, r = np.linalg.qr(sub_trees)
        s = np.linalg.lstsq(q, y_train, rcond=None)[0]
        error = np.divide((s ** 2 * np.sum(q * q, axis=0)), np.sum(out * out, axis=0))

        # If error reduction ration < err_lim, delete root and grow new one
        for i, err in enumerate(error[1:]):
            if err < self.params["error_lim"]:
                del self.roots[i]
                self.grow_tree(
                    max_depth=self.params["max_depth"], method="grow", ind=self
                )

        self.nodes = self.get_sub_nodes()

        num_func_nodes = sum(1 for node in self.nodes if callable(node.value))

        complexity = (
            self.params["complexity_scalar"] * self.total_depth
            + (1 - self.params["complexity_scalar"]) * num_func_nodes
        )

        self.out = out
        self.complexity = complexity
        training_error = None

        if self.params["loss_func"] == "root_mean_square":
            training_error = np.sqrt(np.mean(((y_train - out) ** 2)))

        elif self.params["loss_func"] == "root_median_square":
            training_error = np.sqrt(np.median(((y_train - out) ** 2)))

        return [training_error, self.complexity]
