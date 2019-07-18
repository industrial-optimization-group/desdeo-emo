# BioGP algorithm for pyRVEA, by Niko Rissanen
# For graphics output, install graphviz https://pypi.org/project/graphviz/

from random import random, randint
from graphviz import Digraph, Source
import numpy as np
from inspect import signature
from math import ceil

def add(x, y): return np.add(x, y)
def sub(x, y): return np.subtract(x, y)
def mul(x, y): return np.multiply(x, y)
def div(x, y): return np.divide(x, y)


class BioGP:
    def __init__(self):

        self.pop_size = 60
        self.individuals = []

        self.function_set = [add, sub, mul, div]
        self.terminal_set = ['x', 'y', 0.26, 0.48]   # Matyas

        self.max_depth = 5  # maximal initial random tree depth
        self.prob_terminal = 0.5  # probability to make a node terminal

    def create_individuals(self, method="ramped_half_and_half"):

        if method == "ramped_half_and_half":
            for md in range(ceil(self.max_depth/2), self.max_depth + 1):
                for i in range(int(self.pop_size / self.max_depth + 1)):
                    ind = Node()
                    self.grow_tree(grow=True, max_depth=md, node=ind)  # grow
                    self.individuals.append(ind)
                for i in range(int(self.pop_size / self.max_depth + 1)):
                    ind = Node()
                    self.grow_tree(grow=False, max_depth=md, node=ind)  # full
                    self.individuals.append(ind)

    def grow_tree(self, grow, max_depth, node, depth=0):  # create random tree using either grow or full method
        if depth < 1 or (depth < max_depth and not grow):
            node.value = self.function_set[randint(0, len(self.function_set) - 1)]
        elif depth >= max_depth:
            node.value = self.terminal_set[randint(0, len(self.terminal_set) - 1)]
        else:  # intermediate depth, grow
            if random() > self.prob_terminal:
                node.value = self.terminal_set[randint(0, len(self.terminal_set) - 1)]
            else:
                node.value = self.function_set[randint(0, len(self.function_set) - 1)]
        if node.value in self.function_set:

            for i in range(len(signature(node.value).parameters)):  # Check arity
                root = Node()
                node.roots.append(root)
                self.grow_tree(grow, max_depth, root, depth=depth + 1)


class Node:
    def __init__(self, value=None):

        self.value = value
        self.roots = []

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
        dot[0].attr(kw='graph', label=footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename=name + ".gv", format="png").render()


def main():

    problem = BioGP()
    problem.create_individuals()
    problem.individuals[0].draw_tree("f", "tree0")


if __name__ == "__main__":
    main()
