from pyrvea.Problem.evonn_problem import EvoNNProblem
import numpy as np

input_nodes = 4
hidden_nodes = 6
input_data = np.random.uniform(0, 1, size=(hidden_nodes, input_nodes))

prob = EvoNNProblem(data_input=input_data, num_input_nodes=4, num_hidden_nodes=6)

obj_func = prob.objectives()