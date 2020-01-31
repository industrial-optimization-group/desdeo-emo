import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.surrogatemodels.lipschitzian import LipschitzianRegressor

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs

from desdeo_emo.EAs.RVEA import RVEA, oRVEA, robust_RVEA

from pygmo import non_dominated_front_2d as nd2


problem_names = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
num_var = {"ZDT1": 30, "ZDT2": 30, "ZDT3": 30, "ZDT4": 10, "ZDT6": 10}

for problem_name in problem_names:
    prob = test_problem_builder(problem_name)

    x = lhs(num_var[problem_name], 250)
    y = prob.evaluate(x)

    data_pareto = nd2(y.objectives)
    data_pareto = y.objectives[data_pareto]

    x_names = [f"x{i}" for i in range(1, num_var[problem_name] + 1)]
    y_names = ["f1", "f2"]

    data = pd.DataFrame(np.hstack((x, y.objectives)), columns=x_names + y_names)

    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)

    problem.train(LipschitzianRegressor)
    evolver_L_opt = oRVEA(problem, use_surrogates=True)
    while evolver_L_opt.continue_evolution():
        evolver_L_opt.iterate()

    evolver_L = RVEA(problem, use_surrogates=True)
    while evolver_L.continue_evolution():
        evolver_L.iterate()

    evolver_L_robust = robust_RVEA(problem, use_surrogates=True)
    while evolver_L_robust.continue_evolution():
        evolver_L_robust.iterate()

    problem.train(GaussianProcessRegressor)
    evolver_G_opt = oRVEA(problem, use_surrogates=True)
    while evolver_G_opt.continue_evolution():
        evolver_G_opt.iterate()

    evolver_G = RVEA(problem, use_surrogates=True)
    while evolver_G.continue_evolution():
        evolver_G.iterate()

    evolver_G_robust = robust_RVEA(problem, use_surrogates=True)
    while evolver_G_robust.continue_evolution():
        evolver_G_robust.iterate()

    evolver_true = RVEA(prob)
    while evolver_true.continue_evolution():
        evolver_true.iterate()

    front_true = evolver_true.population.objectives
    front_L = evolver_L.population.objectives
    front_G = evolver_G.population.objectives
    front_L_opt = (
        evolver_L_opt.population.objectives - evolver_L_opt.population.uncertainity
    )
    front_G_opt = (
        evolver_G_opt.population.objectives - evolver_G_opt.population.uncertainity
    )
    front_L_robust = (
        evolver_L_robust.population.objectives
        + evolver_L_robust.population.uncertainity
    )
    front_G_robust = (
        evolver_G_robust.population.objectives
        + evolver_G_robust.population.uncertainity
    )
    plt.clf()
    # Plot 1
    true = plt.scatter(x=front_true[:, 0], y=front_true[:, 1], label="True Front")
    from_data = plt.scatter(
        x=data_pareto[:, 0], y=data_pareto[:, 1], label="Front from data"
    )
    L = plt.scatter(x=front_L[:, 0], y=front_L[:, 1], label="Lipshitzian")
    L_opt = plt.scatter(
        x=front_L_opt[:, 0], y=front_L_opt[:, 1], label="Optimistic Lipschitzian"
    )
    L_robust = plt.scatter(
        x=front_L_robust[:, 0], y=front_L_robust[:, 1], label="Robust Lipshitzian"
    )
    G = plt.scatter(x=front_G[:, 0], y=front_G[:, 1], label="Kriging")
    G_opt = plt.scatter(
        x=front_G_opt[:, 0], y=front_G_opt[:, 1], label="Optimistic Kriging"
    )
    G_robust = plt.scatter(
        x=front_G_robust[:, 0], y=front_G_robust[:, 1], label="Robust Kriging"
    )
    plt.title(
        f"Predicted fronts obtained with various algorithms\n"
        f"on the {problem_name} problem"
    )
    plt.xlabel("F1")
    plt.ylabel("F2")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(
        f"./examples/scripts/optimistic_optimization/" f"{problem_name}_predicted.png"
    )

    front_true = evolver_true.population.objectives
    front_L = prob.evaluate(evolver_L.population.individuals).objectives
    front_G = prob.evaluate(evolver_G.population.individuals).objectives
    front_L_opt = prob.evaluate(evolver_L_opt.population.individuals).objectives
    front_G_opt = prob.evaluate(evolver_G_opt.population.individuals).objectives
    front_L_robust = prob.evaluate(evolver_L_robust.population.individuals).objectives
    front_G_robust = prob.evaluate(evolver_G_robust.population.individuals).objectives

    plt.clf()
    # Plot 2
    true = plt.scatter(x=front_true[:, 0], y=front_true[:, 1], label="True Front")
    from_data = plt.scatter(
        x=data_pareto[:, 0], y=data_pareto[:, 1], label="Front from data"
    )
    L = plt.scatter(x=front_L[:, 0], y=front_L[:, 1], label="Lipshitzian")
    L_opt = plt.scatter(
        x=front_L_opt[:, 0], y=front_L_opt[:, 1], label="Optimistic Lipschitzian"
    )
    L_robust = plt.scatter(
        x=front_L_robust[:, 0], y=front_L_robust[:, 1], label="Robust Lipshitzian"
    )
    G = plt.scatter(x=front_G[:, 0], y=front_G[:, 1], label="Kriging")
    G_opt = plt.scatter(
        x=front_G_opt[:, 0], y=front_G_opt[:, 1], label="Optimistic Kriging"
    )
    G_robust = plt.scatter(
        x=front_G_robust[:, 0], y=front_G_robust[:, 1], label="Robust Kriging"
    )
    plt.title(
        f"Actual fronts obtained with various algorithms\n"
        f"on the {problem_name} problem"
    )
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(
        f"./examples/scripts/optimistic_optimization/" f"{problem_name}_actual.png"
    )
