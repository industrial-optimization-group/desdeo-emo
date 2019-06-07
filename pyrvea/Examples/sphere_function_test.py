from pyrvea.Problem.evonn_problem import EvoNNProblem
from pyrvea.Population.population_evonn import Population
from pyrvea.Problem.testProblem import testProblem
from pyrvea.Recombination.ppga_crossover import ppga_crossover
from pyrvea.Recombination.ppga_mutation import ppga_mutation
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
import pandas
from random import randint, sample
import numpy as np
import cProfile
import timeit
import plotly
import plotly.graph_objs as go
from pyDOE import lhs


def objectives(decision_variables):

    # Sphere function
    # x = np.asarray(decision_variables)
    # obj_func = sum(x ** 2)

    # Matyas function
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = 0.26 * (x**2 + y**2) - 0.48 * x * y

    # Himmelblau's function
    x = np.asarray(decision_variables[0])
    y = np.asarray(decision_variables[1])

    obj_func = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    return obj_func


#rand_training_data_input = np.random.uniform(-10, 10, (150, 2))
lin_training_data_input = np.linspace((-5, -5), (5, 5), 150)
#lin_training_data_input_temp = np.linspace(10, 0, 149)
#x2 = np.zeros_like(lin_training_data_input_temp)
#x2 = np.flip(lin_training_data_input_temp)
#lin_training_data_input = np.vstack((np.hstack((lin_training_data_input_temp,lin_training_data_input_temp)), np.hstack((lin_training_data_input_temp, x2)))).T
#lin_training_data_input = lhs(2, 150) * 20 - 10
#lin_training_data_input[:, 1] = np.flip(lin_training_data_input[:, 1])

training_data_output = np.asarray([objectives(x) for x in lin_training_data_input])

f1 = EvoNNProblem(
    name="EvoNN",
    training_data_input=lin_training_data_input,
    training_data_output=training_data_output,
    num_hidden_nodes=7,
)


def plot(f1):

    popf2 = Population(f1, plotting=False, assign_type="EvoNN")
    model = popf2.evolve(PPGA)
    y_pred = f1.get_prediction(model)

    trace = go.Scatter(x=y_pred, y=training_data_output, mode="markers")

    data = [trace]

    plotly.offline.plot(
        data,
        filename=f1.name
        + "_num_var"
        + str(f1.num_of_variables)
        + "_num_nodes"
        + str(f1.num_hidden_nodes)
        + ".html",
        auto_open=True,
    )


plot(f1)
