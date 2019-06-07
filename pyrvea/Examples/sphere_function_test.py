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

    # Number of data points = 150, number of generations = 100

    # Sphere function, --5 <= x <= 5
    # Error close to zero with random data.
    #
    # x = np.asarray(decision_variables)
    # obj_func = sum(x ** 2)

    # Matyas function, -10 <= x, y <= 10
    # Error close to zero with random data,
    # when number of nodes = 20. With less nodes,
    # training wasn't as successful. With linear data,
    # training wasn't successful.
    #
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = 0.26 * (x**2 + y**2) - 0.48 * x * y

    # Himmelblau's function, -5 <= x, y <= 5
    # Error close to zero with random data.
    #
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    # Rastigrin function, -5.12 <= x <= 5.12
    # Didn't work that well with random data
    # and 2 variables.
    #
    # x = np.asarray(decision_variables)
    # n = len(x)
    # obj_func = 10 * n + sum(x ** 2 - 10 * np.cos(2 * pi * x))

    # Three-hump camel function,  -5 <= x, y <= 5
    # Worked pretty well with random data.
    #
    x = np.asarray(decision_variables[0])
    y = np.asarray(decision_variables[1])
    obj_func = 2*x**2 - 1.05 * x**4 + (x**6)/6 + x * y + y**2

    # Goldstein-Price function, -2 <= x, y <= 2
    #
    # with 15 nodes, min. error == 20000
    # with 20 nodes, min. error == 17000
    # with 25 nodes, min. error == 8671
    #
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * (y ** 2)))\
    #     * (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * (x ** 2) + 48 * y - 36 * x * y + 27 * (y ** 2)))

    # Levi function N.13, -10 <= x, y <= 10
    #
    # with random data and 10 nodes, min. error == 27, not very good
    # with random data and 15 nodes, min. error == 22, bit better
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = (
    #     np.sin(3 * np.pi * x) ** 2
    #     + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
    #     + (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
    # )

    # Schaffer function N. 2, -100 <= x, y <= 100
    #
    # Doesn't work with random data
    # x = np.asarray(decision_variables[0])
    # y = np.asarray(decision_variables[1])
    # obj_func = 0.5 + (np.sin((x**2 - y**2)**2) - 0.5) / (1 + 0.001*(x**2 + y**2))**2

    # Ackley function, -5 <= x, y <= 5
    #

    return obj_func

np.random.seed(42)
rand_training_data_input = np.random.uniform(-5, 5, (150, 2))
# lin_training_data_input = np.linspace((-10, -10), (10, 10), 150)
# lin_training_data_input_temp = np.linspace(10, 0, 149)
# x2 = np.zeros_like(lin_training_data_input_temp)
# x2 = np.flip(lin_training_data_input_temp)
# lin_training_data_input = np.vstack((np.hstack((lin_training_data_input_temp,lin_training_data_input_temp)), np.hstack((lin_training_data_input_temp, x2)))).T
# lin_training_data_input = lhs(2, 150) * 20 - 10
# lin_training_data_input[:, 1] = np.flip(lin_training_data_input[:, 1])

training_data_output = np.asarray([objectives(x) for x in rand_training_data_input])

f1 = EvoNNProblem(
    name="EvoNN",
    training_data_input=rand_training_data_input,
    training_data_output=training_data_output,
    num_hidden_nodes=15,
)


def plot(f1):

    popf2 = Population(f1, plotting=False)
    pr = cProfile.Profile()
    pr.enable()
    model = popf2.evolve(PPGA)
    pr.disable()
    pr.print_stats(sort="cumtime")
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
