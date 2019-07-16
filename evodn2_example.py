from pyrvea.Problem.test_functions import OptTestFunctions
from pyrvea.Problem.dataproblem import DataProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


test_prob = OptTestFunctions("Fonseca-Fleming", num_of_variables=2)
training_data_input, training_data_output = test_prob.create_training_data(
    samples=250, method="random"
)

# # ZDT

# test_prob = testProblem(
#     name="ZDT2",
#     num_of_variables=30,
#     num_of_objectives=2,
#     num_of_constraints=0,
#     upper_limits=1,
#     lower_limits=0,
# )
#
# training_data_input = np.random.rand(250, 30)
# training_data_output = np.asarray(
#     [test_prob.objectives(x) for x in training_data_input]
# )

data = np.hstack((training_data_input, training_data_output))
f1_training_data_output = training_data_output[:, 0]
f2_training_data_output = training_data_output[:, 1]

dataset = pd.DataFrame.from_records(data)
x = []
for n in range(training_data_input.shape[1]):
    x.append("x" + str(n + 1))
y = ["f1", "f2"]
dataset.columns = x + y
problem = DataProblem(data=dataset, x=x, y=y)
problem.train_test_split(train_size=0.7)

problem.train(
    model_type="EvoDN2",
    algorithm=PPGA,
    generations_per_iteration=10,
    iterations=10,
)

# y = problem.models["f1"][0].predict(training_data_input)
# problem.models["f1"][0].plot(y, training_data_output[:, 0], name=test_prob.name + "f1")
#
# y2 = problem.models["f2"][0].predict(training_data_input)
# problem.models["f2"][0].plot(y2, training_data_output[:, 1], name=test_prob.name + "f2")

# problem.train(
#     model_type="EvoNN",
#     algorithm=PPGA
# )
#
# y = problem.models["f1"][2].predict(training_data_input)
# problem.models["f1"][2].plot(y, training_data_output[:, 0], name=test_prob.name + "f1")
#
# y2 = problem.models["f2"][2].predict(training_data_input)
# problem.models["f2"][2].plot(y2, training_data_output[:, 1], name=test_prob.name + "f2")


# Multilayer perceptron
# problem.train(model_type="MLP", max_iter=10000, n_iter_no_change=100)
# mlp_reg_y_pred = problem.models["f1"][0].predict(training_data_input)
#
# trace0 = go.Scatter(x=mlp_reg_y_pred, y=training_data_output[:, 0], mode="markers")
# trace1 = go.Scatter(x=training_data_output[:, 0], y=training_data_output[:, 0])
# data = [trace0, trace1]
# plotly.offline.plot(
#         data,
#         filename="MLP Regressor " + test_prob.name
#                  + ".html",
#         auto_open=True,
# )
#
# mlp_reg_y_pred2 = problem.models["f2"][0].predict(training_data_input)
#
# trace0 = go.Scatter(x=mlp_reg_y_pred2, y=training_data_output[:, 1], mode="markers")
# trace1 = go.Scatter(x=training_data_output[:, 1], y=training_data_output[:, 1])
# data = [trace0, trace1]
# plotly.offline.plot(
#         data,
#         filename="MLP Regressor f2" + test_prob.name
#                  + ".html",
#         auto_open=True,
# )

# y = problem.models["f1"][0].predict(training_data_input)
# problem.models["f1"][0].plot(y, training_data_output[:, 0], name=test_prob.name + "f1")
#
# y2 = problem.models["f2"][0].predict(training_data_input)
# problem.models["f2"][0].plot(y2, training_data_output[:, 1], name=test_prob.name + "f2")

# Optimize
pop_ppga = Population(
    problem,
    pop_size=500,
    assign_type="LHSDesign",
    crossover_type="simulated_binary_crossover",
    mutation_type="bounded_polynomial_mutation",
    plotting=False,
)

pop_rvea = Population(
    problem,
    assign_type="LHSDesign",
    crossover_type="simulated_binary_crossover",
    mutation_type="bounded_polynomial_mutation",
    plotting=False,
)

pop_ppga.evolve(
    PPGA,
    prob_prey_move=0.5,
    prob_mutation=0.1,
    target_pop_size=100,
    kill_interval=4,
    iterations=10,
    generations_per_iteration=10,
)

pop_rvea.evolve(RVEA, iterations=10, generations_per_iteration=25)
pop_ppga.plot_pareto(
    filename="Tests/"
    + problem.models["f1"][0].__class__.__name__
    + "_ppga_"
    + test_prob.name
)
pop_rvea.plot_pareto(
    filename="Tests/"
    + problem.models["f1"][0].__class__.__name__
    + "_rvea_"
    + test_prob.name
)
