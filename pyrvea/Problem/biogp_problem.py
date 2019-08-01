# BioGP algorithm for pyRVEA, by Niko Rissanen
# For graphics output, install graphviz https://pypi.org/project/graphviz/

from random import random, randint, choice, seed
from graphviz import Digraph, Source
import numpy as np
import pandas as pd
from inspect import signature
from math import ceil
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.bioGP import bioGP
from pyrvea.Population.Population import Population
import plotly
import plotly.graph_objs as go


class BioGP(BaseProblem):
    def __init__(
        self,
        name=None,
        X_train=None,
        y_train=None,
        num_of_objectives=2,
        params=None,
        num_samples=None,
        terminal_set=None,
        function_set=None,
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_objectives = num_of_objectives
        self.params = params
        self.num_samples = num_samples
        self.function_map = {
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "sqrt": self.sqrt,
            "log": self.log,
        }
        self.terminal_set = terminal_set
        self.function_set = function_set

        self.individuals = []

    def create_individuals(self, method="ramped_half_and_half"):

        if method == "ramped_half_and_half":
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
        np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)


class BioGPModel(BioGP):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "BioGP_Model"
        self.linear_node = None
        self.fitness = None
        self.svr = None
        self.log = None
        self.set_params(**kwargs)

    def set_params(
        self,
        name="BioGP_Model",
        algorithm=PPGA,
        pop_size=500,
        max_depth=5,
        max_subtrees=3,
        prob_terminal=0.5,
        complexity_scalar=0.5,
        error_lim=0.001,
        selection="min_error",
        recombination_type=None,
        crossover_type="biogp_xover",
        mutation_type="biogp_mut",
        single_obj_generations=5,
        logging=False,
        plotting=False,
        ea_parameters=None,
        function_set=("add", "sub", "mul", "div"),
        terminal_set=None,
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
        ea_parameters : dict
            Contains the parameters needed by EA (Default value = None).
        function_set : iterable
            The function set to use when creating the trees.
        terminal_set : iterable
            The terminals (variables and constants) to use when creating the trees.

        """

        params = {
            "name": name,
            "algorithm": algorithm,
            "pop_size": pop_size,
            "max_depth": max_depth,
            "max_subtrees": max_subtrees,
            "prob_terminal": prob_terminal,
            "complexity_scalar": complexity_scalar,
            "error_lim": error_lim,
            "selection": selection,
            "recombination_type": recombination_type,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "single_obj_generations": single_obj_generations,
            "logging": logging,
            "plotting": plotting,
            "ea_parameters": ea_parameters,
            "function_set": function_set,
            "terminal_set": terminal_set,
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
        self.terminal_set = self.params["terminal_set"]
        function_set = []
        for function in self.params["function_set"]:
            function_set.append(self.function_map[function])
        self.params["function_set"] = function_set

        # if self.params["logging"]:
        #     self.log = self.create_logfile()

        self.train()

        # if self.params["logging"]:
        #     print(self.fitness, file=self.log)

        return self

    def train(self):
        """Trains the networks and selects the best model from the non dominated front.

        """
        pop = Population(
            self,
            assign_type="BioGP",
            pop_size=self.params["pop_size"],
            plotting=self.params["plotting"],
            recombination_type=self.params["recombination_type"],
            crossover_type=self.params["crossover_type"],
            mutation_type=self.params["mutation_type"],
        )

        # Minimize error for first n generations before switching to bi-objective
        ea_params = {
            "generations_per_iteration": self.params["single_obj_generations"],
            "iterations": 1,
        }
        pop.evolve(EA=bioGP, ea_parameters=ea_params)

        pop.evolve(
            EA=self.params["algorithm"], ea_parameters=self.params["ea_parameters"]
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
        y : ndarray
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
            + "_depth"
            + str(self.params["max_depth"])
            + "_subtrees"
            + str(self.params["max_subtrees"])
            + ".html",
            auto_open=True,
        )


class Node:
    """A node object representing a function or terminal node in the tree.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables which are either float or str.
    depth = int
        The depth the node is at.

    """

    def __init__(self, value=None, depth=None, params=None, terminal_set=None, function_set=None):
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
            and random() > self.params["prob_terminal"]
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

        num_func_nodes = sum(
            node.__class__.__name__ == "FunctionNode" for node in self.nodes
        )

        complexity = (
            self.params["complexity_scalar"] * self.total_depth
            + (1 - self.params["complexity_scalar"]) * num_func_nodes
        )

        self.out = out
        self.complexity = complexity
        self.fitness = np.sqrt(((y_train - out) ** 2).mean())

        return [self.fitness, self.complexity]
