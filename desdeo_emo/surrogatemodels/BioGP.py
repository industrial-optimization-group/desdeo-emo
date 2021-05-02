from random import choice, random
from math import ceil
from typing import Callable, Dict, Type, Tuple

import numpy as np
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

from desdeo_emo.EAs.BaseEA import BaseEA
from desdeo_emo.EAs.PPGA import PPGA
from desdeo_emo.EAs.TournamentEA import TournamentEA

# from graphviz import Digraph, Source
import pandas as pd
from desdeo_emo.surrogatemodels.Problem import surrogateProblem
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
from desdeo_emo.recombination.biogp_xover import BioGP_xover
from desdeo_emo.recombination.biogp_mutation import BioGP_mutation
from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_


def negative_r2_score(y_true, y_pred):
    return -r2_score(y_true, y_pred)


class Node:
    """A node object representing a function or terminal node in the tree.

    Parameters
    ----------
    value : function, str or float
        A function node has as its value a function. Terminal nodes contain variables
        which are either float or str.
    depth : int
        The depth the node is at.
    function_set : array_like
        The function set to use when creating the trees.
    terminal_set : array_like
        The terminals (variables and constants) to use when creating the trees.

    """

    def __init__(
        self,
        value,
        depth,
        max_depth,
        max_subtrees,
        prob_terminal,
        function_set,
        terminal_set,
    ):
        self.value = value
        self.depth = depth
        self.max_depth = max_depth
        self.max_subtrees = max_subtrees
        self.prob_terminal = prob_terminal
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

    """def draw_tree(self, name="tree", footer=""):
        dot = [Digraph()]
        dot[0].attr(kw="graph", label=footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename=name + ".gv", format="png").render()"""

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
            max_depth = self.max_depth
        if depth == 0:
            if ind is None:
                ind = LinearNode(value="linear")
            num_subtrees = self.max_subtrees
            for i in range(len(ind.roots), num_subtrees):
                node = self.grow_tree(max_depth, method, depth=depth + 1)
                ind.roots.append(node)

        # Make terminal node
        elif depth >= max_depth or method == "grow" and random() < self.prob_terminal:
            node = Node(
                depth=depth,
                function_set=self.function_set,
                terminal_set=self.terminal_set,
                value=None,
                max_depth=self.max_depth,
                max_subtrees=self.max_subtrees,
                prob_terminal=self.prob_terminal,
            )
            node.value = choice(node.terminal_set)

        # Make function node
        else:
            node = Node(
                depth=depth,
                function_set=self.function_set,
                terminal_set=self.terminal_set,
                value=None,
                max_depth=self.max_depth,
                max_subtrees=self.max_subtrees,
                prob_terminal=self.prob_terminal,
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
    """

    def __init__(
        self,
        max_depth,
        max_subtrees,
        prob_terminal,
        function_set,
        terminal_set,
        error_lim,
        complexity_scalar=0.5,
        value="linear",
        depth=0,
    ):
        super().__init__(
            value=value,
            depth=depth,
            max_depth=max_depth,
            max_subtrees=max_subtrees,
            prob_terminal=prob_terminal,
            function_set=function_set,
            terminal_set=terminal_set,
        )
        self.nodes_at_depth = {}
        self.error_lim = error_lim
        self.complexity_scalar = complexity_scalar
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
            if err < self.error_lim:
                del self.roots[i]
                self.grow_tree(max_depth=self.max_depth, method="grow", ind=self)

        self.nodes = self.get_sub_nodes()

        num_func_nodes = sum(1 for node in self.nodes if callable(node.value))

        complexity = (
            self.complexity_scalar * self.total_depth
            + (1 - self.complexity_scalar) * num_func_nodes
        )

        self.out = out
        self.complexity = complexity

        return [self.out, self.complexity]


class BioGP(BaseRegressor):
    def __init__(
        self,
        training_algorithm: Type[BaseEA] = PPGA,
        pop_size: int = 500,
        probability_crossover: float = 0.9,
        probability_mutation: float = 0.3,
        max_depth: int = 5,
        max_subtrees: int = 4,
        prob_terminal: float = 0.5,
        complexity_scalar: float = 0.5,
        error_lim: float = 0.001,
        init_method: str = "ramped_half_and_half",
        model_selection_criterion: str = "min_error",
        loss_function: str = "mse",
        single_obj_generations: int = 10,
        function_set=("add", "sub", "mul", "div"),
        terminal_set=None,
    ):
        loss_functions = {
            "mse": mean_squared_error,
            "msle": mean_squared_log_error,
            "neg_r2": negative_r2_score,
        }
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
        self.training_algorithm = training_algorithm
        self.pop_size: int = pop_size
        self.probability_crossover: float = probability_crossover
        self.probability_mutation: float = probability_mutation
        self.max_depth: int = max_depth
        self.max_subtrees: int = max_subtrees
        self.prob_terminal: float = prob_terminal
        self.complexity_scalar: float = complexity_scalar
        self.error_lim: float = error_lim
        self.init_method: str = init_method
        self.model_selection_criterion: str = model_selection_criterion
        self.loss_function_str: str = loss_function
        self.loss_function: Callable = loss_functions[loss_function]
        self.single_obj_generations: int = single_obj_generations
        self.function_set_str: Tuple[str] = function_set
        self.function_set = []
        for function in function_set:
            self.function_set.append(self.function_map[function])
        self.terminal_set: Tuple = terminal_set
        #
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.model_trained: bool = False
        # Model Parameters
        self.tree: LinearNode = None
        self.performance: Dict = {"RMSE": None, "R^2": None}
        self.model_population = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if X.shape[0] != y.shape[0]:
            msg = (
                f"Ensure that the number of samples in X and y are the same"
                f"Number of samples in X = {X.shape[0]}"
                f"Number of samples in y = {y.shape[0]}"
            )
            raise ModelError(msg)
        self.X = X
        self.y = y
        if self.terminal_set is None:
            self.terminal_set = X.columns.tolist()

        # Create problem
        problem = surrogateProblem(performance_evaluator=self._model_performance)
        problem.n_of_objectives = 2
        # Create Population
        initial_pop = self._create_individuals()
        population = SurrogatePopulation(
            problem, self.pop_size, initial_pop, None, None, None
        )
        population.xover = BioGP_xover(probability_crossover=self.probability_crossover)
        population.mutation = BioGP_mutation(
            probability_mutation=self.probability_mutation
        )
        # Do single objective evolution
        tournament_evolver = TournamentEA(
            problem,
            initial_population=population,
            population_size=self.pop_size,
            n_gen_per_iter=self.single_obj_generations,
            n_iterations=1,
        )
        figure = animate_init_(
            tournament_evolver.population.objectives, filename="BioGP.html"
        )
        while tournament_evolver.continue_evolution():
            tournament_evolver.iterate()
            figure = animate_next_(
                tournament_evolver.population.objectives,
                figure,
                filename="BioGP.html",
                generation=tournament_evolver._iteration_counter,
            )
        population = tournament_evolver.population

        # Do bi-objective evolution
        evolver = self.training_algorithm(
            problem,
            initial_population=population,
            population_size=self.pop_size,
            n_gen_per_iter=10,
            n_iterations=10,
        )
        while evolver.continue_evolution():
            evolver.iterate()
            figure = animate_next_(
                evolver.population.objectives,
                figure,
                filename="BioGP.html",
                generation=evolver._iteration_counter
                + tournament_evolver._iteration_counter,
            )
        self.model_population = evolver.population
        # Selection
        self.select()
        self.model_trained = True

    def _create_individuals(self):
        individuals = []
        if self.init_method == "ramped_half_and_half":
            for md in range(ceil(self.max_depth / 2), self.max_depth + 1):
                for i in range(int(self.pop_size / (self.max_depth + 1))):

                    ind = LinearNode(
                        value="linear",
                        max_depth=self.max_depth,
                        max_subtrees=self.max_subtrees,
                        prob_terminal=self.prob_terminal,
                        function_set=self.function_set,
                        terminal_set=self.terminal_set,
                        error_lim=self.error_lim,
                    )
                    ind.grow_tree(max_depth=md, method="grow", ind=ind)
                    individuals.append(ind)

                for i in range(int(self.pop_size / (self.max_depth + 1))):

                    ind = LinearNode(
                        value="linear",
                        max_depth=self.max_depth,
                        max_subtrees=self.max_subtrees,
                        prob_terminal=self.prob_terminal,
                        function_set=self.function_set,
                        terminal_set=self.terminal_set,
                        error_lim=self.error_lim,
                    )
                    ind.grow_tree(max_depth=md, method="full", ind=ind)
                    individuals.append(ind)

        elif self.init_method == "full":
            for i in range(self.pop_size):
                ind = LinearNode(
                    value="linear",
                    max_depth=self.max_depth,
                    max_subtrees=self.max_subtrees,
                    prob_terminal=self.prob_terminal,
                    function_set=self.function_set,
                    terminal_set=self.terminal_set,
                    error_lim=self.error_lim,
                )
                ind.grow_tree(method="full", ind=ind)
                individuals.append(ind)

        elif self.init_method == "grow":
            for i in range(self.pop_size):
                ind = LinearNode(
                    value="linear",
                    max_depth=self.max_depth,
                    max_subtrees=self.max_subtrees,
                    prob_terminal=self.prob_terminal,
                    function_set=self.function_set,
                    terminal_set=self.terminal_set,
                    error_lim=self.error_lim,
                )
                ind.grow_tree(method="grow", ind=ind)
                individuals.append(ind)

        return np.asarray(individuals).reshape(-1, 1)

    def _model_performance(
        self, trees: LinearNode, X: np.ndarray = None, y: np.ndarray = None
    ):
        if trees is None and self.model_trained is False:
            msg = "Model has not been trained yet"
            raise ModelError(msg)
        if trees is None:
            trees = self.tree
        if X is None:
            X = self.X
            y = self.y
        if len(trees) > 1:
            loss = []
            complexity = []
            for tree in trees:
                y_pred, ind_complexity = tree[0].calculate_linear(X, y)
                loss.append(self.loss_function(y, y_pred))
                complexity.append(ind_complexity)
        else:
            y_pred, complexity = trees[0].calculate_linear(X, y)
            loss = self.loss_function(y, y_pred)
        return np.asarray((loss, complexity)).T

    def predict(self, X: np.ndarray):
        if isinstance(X, (np.ndarray, list)):
            X = pd.DataFrame(X, columns=self.X.columns)
        sub_trees = []
        # Stack outputs of subtrees to form weight matrix
        for root in self.tree.roots:
            sub_tree = root.predict(X)
            if sub_tree.size == 1:
                sub_tree = sub_tree.reshape(1)
            sub_trees.append(sub_tree)

        sub_trees = np.hstack(sub_trees)
        axis = None
        if sub_trees.ndim > 1:
            axis = 1
        sub_trees = np.insert(sub_trees, 0, 1, axis=axis)

        y = np.dot(sub_trees, self.tree.linear)

        if isinstance(y, float):
            y = np.asarray([y])

        return y, np.zeros_like(y)

    def select(self):
        if self.model_selection_criterion == "min_error":
            # Return the model with the lowest error
            selected = np.argmin(self.model_population.objectives[:, 0])
            # print(self.model_population.objectives)
        else:
            raise ModelError("Selection criterion not recognized. Use 'min_error'.")
        self.tree = self.model_population.individuals[selected][0]
        y_pred = self.predict(X=self.X)[0]
        self.performance["RMSE"] = np.sqrt(mean_squared_error(self.y, y_pred))
        self.performance["R^2"] = r2_score(self.y, y_pred)

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
