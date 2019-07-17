from pyrvea.Problem.test_functions import OptTestFunctions
from pyrvea.Problem.dataproblem import DataProblem
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go

test_prob = TestProblem(name="Fonseca-Fleming", num_of_variables=2, num_of_objectives=2)

dataset, x, y = test_prob.create_training_data(samples=500, method="lhs")

problem = DataProblem(data=dataset, x=x, y=y)
problem.train_test_split(train_size=0.7)
ea_parameters = {
    "target_pop_size": 50,
    "generations_per_iteration": 10,
    "iterations": 10,
}
problem.train(model_type="EvoNN", algorithm=PPGA, ea_parameters=ea_parameters)

y = problem.models["f1"][0].predict(np.asarray(problem.data[problem.x]))
problem.models["f1"][0].plot(
    y, np.asarray(problem.data["f1"]), name=test_prob.name + "f1"
)

y2 = problem.models["f2"][0].predict(np.asarray(problem.data[problem.x]))
problem.models["f2"][0].plot(
    y2, np.asarray(problem.data["f2"]), name=test_prob.name + "f2"
)

# problem.train(
#     model_type="EvoNN",
#     algorithm=PPGA
# )
#
# y = problem.models["f1"][0].predict(training_data_input)
# problem.models["f1"][0].plot(y, training_data_output[:, 0], name=test_prob.name + "f1")
#
# y2 = problem.models["f2"][0].predict(training_data_input)
# problem.models["f2"][0].plot(y2, training_data_output[:, 1], name=test_prob.name + "f2")


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

ppga_params = {
    "prob_prey_move": 0.5,
    "prob_mutation": 0.1,
    "target_pop_size": 100,
    "kill_interval": 4,
    "iterations": 10,
    "generations_per_iteration": 10,
}

pop_ppga.evolve(PPGA, ea_parameters=ppga_params)

pop_ppga.plot_pareto(
    name="Tests/"
    + problem.models["f1"][0].__class__.__name__
    + "_ppga_"
    + test_prob.name
)

pop_rvea = Population(
    problem,
    assign_type="LHSDesign",
    crossover_type="simulated_binary_crossover",
    mutation_type="bounded_polynomial_mutation",
    plotting=False,
)

rvea_params = {"iterations": 10, "generations_per_iteration": 25}

pop_rvea.evolve(RVEA, ea_parameters=rvea_params)

pop_rvea.plot_pareto(
    name="Tests/"
    + problem.models["f1"][0].__class__.__name__
    + "_rvea_"
    + test_prob.name
)
