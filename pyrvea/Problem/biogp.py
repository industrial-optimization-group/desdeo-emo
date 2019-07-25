# BioGP algorithm for pyRVEA, by Niko Rissanen
# For graphics output, install graphviz https://pypi.org/project/graphviz/

from random import random, randint, choice, seed
from graphviz import Digraph, Source
import numpy as np
from inspect import signature
from math import ceil
from pyrvea.Problem.testproblem import TestProblem


class BioGP:
    def __init__(self):

        self.pop_size = 60
        self.individuals = []

        self.max_depth = 5  # maximum initial random tree depth
        self.max_subtrees = 5
        self.prob_terminal = 0.5  # probability to make a node terminal
        self.total_func_nodes = 0

    def create_individuals(self, method="ramped_half_and_half"):

        if method == "ramped_half_and_half":
            for md in range(ceil(self.max_depth / 2), self.max_depth + 1):
                for i in range(int(self.pop_size / (self.max_depth + 1))):
                    self.total_func_nodes = 0
                    self.grow_tree(max_depth=md, method="grow")

                for i in range(int(self.pop_size / (self.max_depth + 1))):
                    self.total_func_nodes = 0
                    self.grow_tree(max_depth=md, method="full")

    def grow_tree(self, max_depth, method="grow", depth=0):
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
            ind = LinearNode(value="linear")
            for i in range(self.max_subtrees):
                node = self.grow_tree(max_depth, method, depth=depth + 1)
                ind.roots.append(node)
            ind.num_func_nodes = self.total_func_nodes
            self.individuals.append(ind)

        # Make terminal node
        elif depth >= max_depth or method == "grow" and random() > self.prob_terminal:
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
        self.value = value
        self.depth = depth
        self.scalar = scalar
        self.num_func_nodes = 0
        super().__init__()

    def calculate_linear(self, decision_variables, y_train):

        sub_trees = []
        for root in self.roots:
            sub_trees.append([root.predict(x) for x in decision_variables])

        fx = np.insert(np.swapaxes(np.asarray(sub_trees), 0, 1), 0, 1, axis=1)

        w, *_ = np.linalg.lstsq(fx, y_train, rcond=None)

        y = np.dot(fx, w)

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

        complexity = (
            self.scalar * self.depth_count(self)
            + (1 - self.scalar) * self.num_func_nodes
        )

        return y, complexity


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


def main():

    # seed(1)

    problem = BioGP()
    problem.create_individuals()
    problem.individuals[0].draw_tree("f", "tree0")

    test_prob = TestProblem("Matyas", num_of_variables=2, num_of_objectives=1)
    dataset, x, y = test_prob.create_training_data(3)
    X_train = np.asarray(dataset[x])
    y_train = np.asarray(dataset[y])
    out = problem.individuals[0].calculate_linear(X_train, y_train)
    # out = [problem.individuals[0].predict(x) for x in X_train]
    # out = np.dot(dataset, problem.individuals[0].predict())
    return


if __name__ == "__main__":
    main()
