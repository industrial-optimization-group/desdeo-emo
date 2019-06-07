from pyrvea.Problem.evonn_problem import EvoNNProblem
from pyrvea.Population.population_evonn import Population
from pyrvea.Problem.testProblem import testProblem
from pyrvea.Recombination.ppga_crossover import ppga_crossover
from pyrvea.Recombination.ppga_mutation import ppga_mutation
from pyrvea.EAs.PPGA import PPGA
import pandas
from random import randint, sample
import numpy as np
import cProfile
import timeit
import plotly
import plotly.graph_objs as go

zdt_problem = testProblem('ZDT2', 30, 2, 0, 1, 0)
training_data_input = np.random.rand(150,30)
training_data_output = np.asarray([zdt_problem.objectives(x) for x in training_data_input])

f1_training_data_output = training_data_output[:, 0]
f2_training_data_output = training_data_output[:, 1]

#preferred_output = np.random.uniform(0,1,num_of_samples)

def plot(f2):

    popf2 = Population(f2, plotting=False)
    model = popf2.evolve(PPGA)
    y_pred = f2.get_prediction(model)

    trace = go.Scatter(
        x=y_pred,
        y=f2_training_data_output,
        mode='markers'
    )

    data = [trace]

    plotly.offline.plot(data, filename=f2.name + ".html", auto_open=True)


# f1 = EvoNNProblem(
#     name="EvoNN",
#     training_data_input=training_data_input,
#     training_data_output=f1_training_data_output,
# )

# f2 = EvoNNProblem(
#     name="EvoNN_nodes_10",
#     training_data_input=training_data_input,
#     training_data_output=f2_training_data_output,
#     num_hidden_nodes=10
# )
#
# plot(f2)

f2 = EvoNNProblem(
    name="ZDT2_15_nodes",
    training_data_input=training_data_input,
    training_data_output=f2_training_data_output,
    num_hidden_nodes=15
)

plot(f2)

#arr = np.zeros((60,60))

#popf1 = Population(f1, plotting=True)
#popf2 = Population(f2, plotting=True)

#popf1.evolve(PPGA)

# Profiling

# pr = cProfile.Profile()
# pr.enable()
# model = popf2.evolve(PPGA)
# pr.disable()
# pr.print_stats(sort="cumtime")

#non_dom_front = nds(popf1.objectives)
#print("a")
#p = PPGA(pop)

#a1 = randint(0, 49)
#a2 = randint(0, 49)
#mutation(pop, a1, a2)

#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.keep(selected)
#selected = sample(range(1, np.shape(pop.individuals)[0]), randint(1, np.shape(pop.individuals)[0]))
#pop.delete(selected)