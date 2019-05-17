from pyrvea.Problem.evonn_problem import EvoNNProblem

import numpy as np

input_nodes = 4
hidden_nodes = 6
training_data = np.random.uniform(0, 1, size=(hidden_nodes, input_nodes))

prob = EvoNNProblem(training_data_input=training_data, num_input_nodes=4, num_hidden_nodes=6)

obj_func = prob.objectives()

