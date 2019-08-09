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

test_prob = TestProblem(name="Fonseca-Fleming", num_of_variables=2)
dataset, x, y = test_prob.create_training_data(samples=500)

# dataset = pd.read_excel("ZDT1_1000.xls")
# x = dataset.columns[0:30].tolist()
# y = dataset.columns[30:].tolist()

problem = DataProblem(data=dataset, x=x, y=y)
problem.train_test_split(train_size=0.7)

f_set = ("add", "sub", "mul", "div", "sqrt", "neg")
t_set = [1, 2]
problem.train(
    model_type="BioGP",
    algorithm=RVEA,
    terminal_set=t_set,
    function_set=f_set,
)

# problem.train(
#     model_type="EvoDN2",
#     algorithm=RVEA,
# )

y = problem.models["f1"][0].predict(problem.data[problem.x])
problem.models["f1"][0].plot(y, problem.data["f1"], name="ZDT1_100" + "f1")

y2 = problem.models["f2"][0].predict(problem.data[problem.x])
problem.models["f2"][0].plot(y2, problem.data["f2"], name="ZDT1_100" + "f2")

# Optimize
# PPGA
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
    "prob_mutation": 0.5,
    "target_pop_size": 100,
    "kill_interval": 4,
    "iterations": 10,
    "predator_pop_size": 60,
    "generations_per_iteration": 10,
}

pop_ppga.evolve(PPGA, ea_parameters=ppga_params)

pop_ppga.plot_pareto(
    name="Tests/" + problem.models["f1"][0].__class__.__name__ + "_ppga_" + "ZDT1_100"
)

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