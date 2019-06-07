from pygmo import fast_non_dominated_sorting as nds
from pyrvea.Problem.evonn_problem import EvoNNProblem
from pyrvea.Population.population_evonn import Population
from pyrvea.Problem.testProblem import testProblem
from pyrvea.Recombination.ppga_crossover import ppga_crossover
from pyrvea.Recombination.ppga_mutation import ppga_mutation
from pyrvea.EAs.PPGA import PPGA
import pandas
from random import randint, sample
import numpy as np
import timeit
import plotly
import plotly.graph_objs as go

test_data = pandas.read_csv("178deb.csv", header=1)

#training_data = np.random.uniform(0, 1, size=(num_of_samples, input_nodes))
training_data = test_data.values[:, :-2]
f1_training_data_output = test_data.values[:, 2]
f2_training_data_output = test_data.values[:, 3]
input_nodes = training_data.shape[1]
#preferred_output = np.random.uniform(0,1,num_of_samples)

f1 = EvoNNProblem(
    name="EvoNN",
    training_data_input=training_data,
    training_data_output=f1_training_data_output,
    num_input_nodes=input_nodes,
)

f2 = EvoNNProblem(
    name="EvoNN",
    training_data_input=training_data,
    training_data_output=f2_training_data_output,
    num_input_nodes=input_nodes,
)

#arr = np.zeros((60,60))

popf1 = Population(f1, plotting=False)
# popf2 = Population(f2)

model = popf1.evolve(PPGA)
# popf2.evolve(PPGA)

y_pred = f1.get_prediction(model)

trace = go.Scatter(
    x = y_pred,
    y = f1_training_data_output,
    mode = 'markers'
)

data = [trace]

plotly.offline.plot(data, filename='temp-plot.html', auto_open=True)
#p = PPGA(pop)

#a1 = randint(0, 49)
#a2 = randint(0, 49)
#mutation(pop, a1, a2)

#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.keep(selected)
#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.delete(selected)