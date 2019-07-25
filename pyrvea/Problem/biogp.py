# BioGP algorithm for pyRVEA, by Niko Rissanen
# For graphics output, install graphviz https://pypi.org/project/graphviz/

from random import random, randint, choice, seed
from graphviz import Digraph, Source
import numpy as np
from inspect import signature
from math import ceil
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.EAs.PPGA import PPGA
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
    ):
        super().__init__()

        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_objectives = num_of_objectives
        self.params = params
        self.num_samples = num_samples

        self.individuals = []

        self.total_func_nodes = 0

    def create_individuals(self, method="ramped_half_and_half"):

        if method == "ramped_half_and_half":
            for md in range(ceil(self.params["max_depth"] / 2), self.params["max_depth"] + 1):
                for i in range(int(self.params["pop_size"] / (self.params["max_depth"] + 1))):
                    self.total_func_nodes = 0
                    self.grow_tree(max_depth=md, method="grow")

                for i in range(int(self.params["pop_size"]/ (self.params["max_depth"] + 1))):
                    self.total_func_nodes = 0
                    self.grow_tree(max_depth=md, method="full")

        return self.individuals

    def grow_tree(self, max_depth, method="grow", depth=0, ind=None):
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

        """
        node = None

        if depth == 0:
            if ind is None:
                ind = LinearNode(value="linear")
            for i in range(len(ind.roots), self.params["max_subtrees"]):
                node = self.grow_tree(max_depth, method, depth=depth + 1)
                ind.roots.append(node)
            ind.num_func_nodes = self.total_func_nodes
            self.individuals.append(ind)

        # Make terminal node
        elif depth >= max_depth or method == "grow" and random() > self.params["prob_terminal"]:
            node = TerminalNode(depth=depth)
            node.value = choice(node.terminal_set)

        # Make function node
        else:
            node = FunctionNode(depth=depth)
            self.total_func_nodes += 1
            node.value = choice(list(node.function_set.values()))
            for i in range(node.value.__code__.co_argcount):  # Check arity
                root = self.grow_tree(max_depth, method, depth=depth + 1)
                node.roots.append(root)

        return node

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


class BioGPModel(BioGP):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "BioGP_Model"
        self.linear_node = None
        self.fitness = None
        self.set_params(**kwargs)

    def set_params(
        self,
        name="BioGP_Model",
        algorithm=PPGA,
        pop_size=500,
        max_depth=6,
        max_subtrees=5,
        prob_terminal=0.5,
        selection="min_error",
        recombination_type=None,
        crossover_type="biogp_xover_standard",
        mutation_type="biogp_mut_standard",
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
            "max_depth": max_depth,
            "max_subtrees": max_subtrees,
            "prob_terminal": prob_terminal,
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
        pop.evolve(
            EA=self.params["algorithm"],
            ea_parameters=self.params["ea_parameters"]
        )

        non_dom_front = pop.non_dominated()
        self.linear_node, self.fitness = self.select(
            pop, non_dom_front, self.params["selection"]
        )

    def predict(self, decision_variables):

        sub_trees = []
        for root in self.linear_node.roots:
            sub_trees.append([root.predict(x) for x in decision_variables])

        f_matrix = np.insert(np.swapaxes(np.asarray(sub_trees), 0, 1), 0, 1, axis=1)

        out = np.dot(f_matrix, self.linear_node.linear)

        return out

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

    def __init__(self, value=None, depth=None):

        self.value = value
        self.depth = depth
        self.roots = []

    def predict(self, decision_variables=None):
        pass

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

    def draw_tree(self, name, footer):
        dot = [Digraph()]
        dot[0].attr(kw="graph", label=footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename=name + ".gv", format="png").render()

    def depth_count(self, node):

        if isinstance(node.roots, list):
            if len(node.roots) == 0:
                depth = 1
            else:
                depth = 1 + max([self.depth_count(node) for node in node.roots])
        else:
            depth = 0
        return depth


class LinearNode(Node):
    def __init__(self, value="linear", depth=0, scalar=0.5):
        super().__init__()
        self.value = value
        self.depth = depth
        self.scalar = scalar
        self.num_func_nodes = 0
        self.out = None
        self.complexity = None
        self.fitness = None
        self.linear = None

    def calculate_linear(self, X_train, y_train):

        sub_trees = []
        for root in self.roots:
            sub_trees.append([root.predict(x) for x in X_train])

        f_matrix = np.insert(np.swapaxes(np.asarray(sub_trees), 0, 1), 0, 1, axis=1)

        weights, *_ = np.linalg.lstsq(f_matrix, y_train, rcond=None)
        self.linear = weights
        out = np.dot(f_matrix, weights)

        # Error reduction ratio
        #
        # q, r = np.linalg.qr(fx)
        #
        # d_inv = np.linalg.inv(np.matmul(q, q.T))
        #
        # #s = np.matmul(d_inv, np.matmul(q.T, y))
        # s = np.linalg.lstsq(q, y_train)[0]
        #
        # error = np.divide((s**2 * np.matmul(q.T, q)), np.matmul(y.T, y))

        self.depth = self.depth_count(self)

        complexity = (
            self.scalar * self.depth
            + (1 - self.scalar) * self.num_func_nodes
        )

        self.out = out
        self.complexity = complexity
        self.fitness = np.sqrt(((y_train - out) ** 2).mean())

        return [self.fitness, self.complexity]


class FunctionNode(Node):
    def __init__(self, value=None, depth=None):
        self.value = value
        self.depth = depth
        self.function_set = {
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
        }
        super().__init__()

    def predict(self, decision_variables=None):
        dv = decision_variables

        values = [root.predict(dv) for root in self.roots]
        if self.depth == 0:
            self.value = sum(values)
            return self.value
        else:
            return self.value(*values)

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
        if -0.001 <= y <= 0.001:
            return 1.0
        else:
            return np.divide(x, y)


class TerminalNode(Node):
    def __init__(self, value=None, depth=None):
        self.value = value
        self.depth = depth
        self.terminal_set = ["x1", "x2", 0.26, 0.48]
        super().__init__()

    def predict(self, decision_variables=None):

        if self.value == "x1":
            return decision_variables[0]
        elif self.value == "x2":
            return decision_variables[1]
        # if isinstance(self.value, str):
        #     return decision_variables[self.value]

        else:
            return self.value


# def main():
#
#     # seed(1)
#
#     problem = BioGP()
#     problem.create_individuals()
#     problem.individuals[0].draw_tree("f", "tree0")
#
#     test_prob = TestProblem("Matyas", num_of_variables=2, num_of_objectives=1)
#     dataset, x, y = test_prob.create_training_data(3)
#     X_train = np.asarray(dataset[x])
#     y_train = np.asarray(dataset[y])
#     out = problem.individuals[0].calculate_linear(X_train, y_train)
#     # out = [problem.individuals[0].predict(x) for x in X_train]
#     # out = np.dot(dataset, problem.individuals[0].predict())
#     return
#
#
# if __name__ == "__main__":
#     main()
