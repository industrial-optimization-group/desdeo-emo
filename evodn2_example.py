from pyrvea.Problem.dataproblem import DataProblem
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
import numpy as np
import pandas as pd
from copy import deepcopy

# test_prob = TestProblem(name="ZDT1", num_of_variables=30, num_of_objectives=2)
# dataset, x, y = test_prob.create_training_data(samples=500, seed=10)

dataset = pd.read_excel("ZDT1_1000.xls")
x = []
y = []
for var in range(30):
    x.append("x" + str(var + 1))
for obj in range(2):
    y.append("f" + str(obj + 1))
dataset.columns = x + y

problem = DataProblem(data=dataset, x=x, y=y)
problem.train_test_split(train_size=0.7)

ea_params = {
    "generations_per_iteration": 10,
    "iterations": 10,
    "predator_pop_size": 60,
    "target_pop_size": 300,
    "prob_prey_move": 0.5,
    "prob_crossover": 0.9,
    "prob_mutation": 0.3,
    "kill_interval": 5,
    "max_rank": 20,
}

f_set = ("add", "sub", "mul", "div", "sqrt")
t_set = deepcopy(x)
t_set.extend([1, 2, 9, 29])
# problem.train(
#     model_type="BioGP",
#     algorithm=PPGA,
#     ea_parameters=ea_params,
#     terminal_set=t_set,
#     function_set=f_set,
#     logging=True,
#     single_obj_generations=10
# )

ea_params_evonn = {
    "generations_per_iteration": 10,
    "iterations": 10,
    "prob_crossover": 0.8,
    "prob_mutation": 0.3,
    "predator_pop_size": 60,
    "target_pop_size": 100,
}

problem.train(
    model_type="EvoNN",
    algorithm=RVEA,
    selection="min_error",
    activation_func="sigmoid",
    ea_parameters=ea_params_evonn,
)

y = problem.models["f1"][0].predict(problem.data[problem.x])
problem.models["f1"][0].plot(y, np.asarray(problem.data["f1"]), name="ZDT1_100" + "f1")

y2 = problem.models["f2"][0].predict(problem.data[problem.x])
problem.models["f2"][0].plot(y2, np.asarray(problem.data["f2"]), name="ZDT1_100" + "f2")

# Optimize
# PPGA
# pop_ppga = Population(
#     problem,
#     pop_size=500,
#     assign_type="LHSDesign",
#     crossover_type="simulated_binary_crossover",
#     mutation_type="bounded_polynomial_mutation",
#     plotting=False,
# )
#
# ppga_params = {
#     "prob_prey_move": 0.5,
#     "prob_mutation": 0.5,
#     "target_pop_size": 100,
#     "kill_interval": 4,
#     "iterations": 10,
#     "predator_pop_size": 60,
#     "generations_per_iteration": 10,
# }
#
# pop_ppga.evolve(PPGA, ea_parameters=ppga_params)
#
# pop_ppga.plot_pareto(
#     name="Tests/" + problem.models["f1"][0].__class__.__name__ + "_ppga_" + "ZDT1_100"
# )

# RVEA
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
    name="Tests/" + problem.models["f1"][0].__class__.__name__ + "_rvea_" + "ZDT1_100"
)