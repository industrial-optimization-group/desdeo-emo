from pygmo import fast_non_dominated_sorting as nds
from pyrvea.Problem.evonn_problem import EvoNNProblem
from pyrvea.Population.population_evonn import Population
from pyrvea.Recombination.ppga_crossover import ppga_crossover
from pyrvea.Recombination.ppga_mutation import ppga_mutation
from pyrvea.EAs.PPGA import PPGA
from pandas import read_csv
from random import randint, sample
import numpy as np
import timeit

test_data = read_csv("178deb.csv", header=1)

#training_data = np.random.uniform(0, 1, size=(num_of_samples, input_nodes))
training_data = test_data.values[:, :-2]
training_data_output = test_data.values[:, 2:]
input_nodes = training_data.shape[1]
#preferred_output = np.random.uniform(0,1,num_of_samples)

prob = EvoNNProblem(
    name="EvoNN",
    training_data_input=training_data,
    training_data_output=training_data_output,
    num_input_nodes=input_nodes,
)

#arr = np.zeros((60,60))

pop = Population(prob)
pop.evolve(PPGA)
non_dom_front = nds(pop.objectives)
print("a")
#p = PPGA(pop)

#a1 = randint(0, 49)
#a2 = randint(0, 49)
#mutation(pop, a1, a2)

#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.keep(selected)
#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.delete(selected)