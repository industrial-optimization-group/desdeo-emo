from pyrvea.Problem.evonn_problem import EvoNNProblem
from pyrvea.Population.population_evonn import PopulationEvoNN
from random import randint, sample

import numpy as np

input_nodes = 5
hidden_nodes = 4
num_of_samples = 10
training_data = np.random.uniform(0, 1, size=(num_of_samples, input_nodes))
preferred_output = np.ones(num_of_samples)

prob = EvoNNProblem(
    name="EvoNN",
    training_data_input=training_data,
    training_data_output=preferred_output,
    num_input_nodes=input_nodes,
    num_hidden_nodes=hidden_nodes,
)

pop = PopulationEvoNN(prob)

selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.keep(selected)
selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
pop.delete(selected)