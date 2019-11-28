import numpy as np
from random import choice, random
from graphviz import Digraph, Source
from desdeo_problem.surrogatemodels.SurrogateModels import (BaseRegressor,
                                                            ModelError)


class Node:
    """A node object representing a function or terminal node in the tree.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables
        which are either float or str.
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
            For the 'grow' method, nodes are chosen at random from both functions and
            terminals.
            The 'full' method chooses nodes from the function set until the max depth
            is reached,
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
    by the user. The linear node takes a weighted sum of the output from the subtrees
    and
    also uses a bias value. The weights and the bias are calculated by the linear least
    square technique.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables
        which are either float or str.
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

class BioGP(BaseRegressor):
    def __init__(self,)