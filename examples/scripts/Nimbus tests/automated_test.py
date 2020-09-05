import numpy as np
import pandas as pd

from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs.NIMBUS_EA import NIMBUS_RVEA, NIMBUS_NSGAIII
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from desdeo_tools.scalarization.ASF import PointMethodASF as asf

from tqdm import tqdm


"""def get_min_scalar(scalar, response, ideal, nadir):
    def obj(y):
        if np.any(y < 0):
            return 100
        if np.linalg.norm(y) < 1:
            return 100
        else:
            return scalar(y, response)

    y = differential_evolution(obj, bounds=np.asarray((ideal, nadir)).T).x
    return scalar(y, response)"""


problem_names = ["DTLZ2", "DTLZ3", "DTLZ4"]

n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])
K = 10
n_vars = K + n_objs - 1

num_gen_per_iter = [100, 150, 200, 250]

algorithms = ["RVEA", "NSGAIII", "iRVEA", "iNSGAIII", "NIMBUS-RVEA", "NIMBUS-NSGAIII"]
column_names = (
    ["problem", "num_obj", "iteration", "num_gens"]
    + [algorithm + "_median_asf" for algorithm in algorithms]
    + [algorithm + "_min_asf" for algorithm in algorithms]
    + [algorithm + "_25q_asf" for algorithm in algorithms]
    + [algorithm + "_75q_asf" for algorithm in algorithms]
    + [algorithm + "_max_asf" for algorithm in algorithms]
    + [algorithm + "_median_norm" for algorithm in algorithms]
    + [algorithm + "_min_norm" for algorithm in algorithms]
    + [algorithm + "_25q_norm" for algorithm in algorithms]
    + [algorithm + "_75q_norm" for algorithm in algorithms]
    + [algorithm + "_max_norm" for algorithm in algorithms]
    + [algorithm + "_num_solns" for algorithm in algorithms]
    + [algorithm + "_num_func_eval" for algorithm in algorithms]
)

excess_columns = [
    "_median_asf",
    "_min_asf",
    "_25q_asf",
    "_75q_asf",
    "_max_asf",
    "_median_norm",
    "_min_norm",
    "_25q_norm",
    "_75q_norm",
    "_max_norm",
    "_num_solns",
    "_num_func_eval",
]

data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for gen in num_gen_per_iter:
    for n_obj, n_var in zip(n_objs, n_vars):
        for problem_name in problem_names:
            print(f"Loop {counter} of {total_count}")
            counter += 1
            problem = test_problem_builder(
                name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
            )
            problem.ideal = np.asarray([0] * n_obj)
            problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

            true_nadir = np.asarray([1] * n_obj)

            scalar = asf(ideal=problem.ideal, nadir=true_nadir)

            # a posteriori
            a_post_rvea = RVEA(
                problem=problem, interact=False, n_gen_per_iter=gen, n_iterations=4
            )
            a_post_nsga = NSGAIII(
                problem=problem, interact=False, n_gen_per_iter=gen, n_iterations=4
            )
            # interactive
            int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
            int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)
            # New algorithm
            nimb_rvea = NIMBUS_RVEA(problem, n_gen_per_iter=gen)
            nimb_nsga = NIMBUS_NSGAIII(problem, n_gen_per_iter=gen)

            responses = np.random.rand(4, n_obj)

            _, pref_int_rvea = int_rvea.requests()
            _, pref_int_nsga = int_nsga.requests()
            _, pref_nimb_rvea = nimb_rvea.requests()
            _, pref_nimb_nsga = nimb_nsga.requests()

            for i, response in enumerate(responses):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]

                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_nsga.response = pd.DataFrame(
                    [response], columns=pref_int_nsga.content["dimensions_data"].columns
                )
                pref_nimb_rvea.response = pd.DataFrame(
                    [response],
                    columns=pref_nimb_rvea.content["dimensions_data"].columns,
                )
                pref_nimb_nsga.response = pd.DataFrame(
                    [response],
                    columns=pref_nimb_nsga.content["dimensions_data"].columns,
                )

                a_post_rvea.iterate()
                a_post_nsga.iterate()
                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)
                _, pref_nimb_rvea = nimb_rvea.iterate(pref_nimb_rvea)
                _, pref_nimb_nsga = nimb_nsga.iterate(pref_nimb_nsga)

                scalar_rvea = scalar(a_post_rvea.population.objectives, response)
                scalar_nsga = scalar(a_post_nsga.population.objectives, response)
                scalar_irvea = scalar(int_rvea.population.objectives, response)
                scalar_insga = scalar(int_nsga.population.objectives, response)
                scalar_nrvea = scalar(nimb_rvea.population.objectives, response)
                scalar_nnsga = scalar(nimb_nsga.population.objectives, response)

                norm_rvea = np.linalg.norm(a_post_rvea.population.objectives, axis=1)
                norm_nsga = np.linalg.norm(a_post_nsga.population.objectives, axis=1)
                norm_irvea = np.linalg.norm(int_rvea.population.objectives, axis=1)
                norm_insga = np.linalg.norm(int_nsga.population.objectives, axis=1)
                norm_nrvea = np.linalg.norm(nimb_rvea.population.objectives, axis=1)
                norm_nnsga = np.linalg.norm(nimb_nsga.population.objectives, axis=1)

                data_row[["RVEA" + excess_col for excess_col in excess_columns]] = [
                    np.median(scalar_rvea),
                    np.min(scalar_rvea),
                    np.quantile(scalar_rvea, 0.25),
                    np.quantile(scalar_rvea, 0.75),
                    np.max(scalar_rvea),
                    np.median(norm_rvea),
                    np.min(norm_rvea),
                    np.quantile(norm_rvea, 0.25),
                    np.quantile(norm_rvea, 0.75),
                    np.max(norm_rvea),
                    a_post_rvea.population.objectives.shape[0],
                    a_post_rvea._function_evaluation_count,
                ]
                data_row[["NSGAIII" + excess_col for excess_col in excess_columns]] = [
                    np.median(scalar_nsga),
                    np.min(scalar_nsga),
                    np.quantile(scalar_nsga, 0.25),
                    np.quantile(scalar_nsga, 0.75),
                    np.max(scalar_nsga),
                    np.median(norm_nsga),
                    np.min(norm_nsga),
                    np.quantile(norm_nsga, 0.25),
                    np.quantile(norm_nsga, 0.75),
                    np.max(norm_nsga),
                    a_post_nsga.population.objectives.shape[0],
                    a_post_nsga._function_evaluation_count,
                ]
                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    np.median(scalar_irvea),
                    np.min(scalar_irvea),
                    np.quantile(scalar_irvea, 0.25),
                    np.quantile(scalar_irvea, 0.75),
                    np.max(scalar_irvea),
                    np.median(norm_irvea),
                    np.min(norm_irvea),
                    np.quantile(norm_irvea, 0.25),
                    np.quantile(norm_irvea, 0.75),
                    np.max(norm_irvea),
                    int_rvea.population.objectives.shape[0],
                    int_rvea._function_evaluation_count,
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    np.median(scalar_insga),
                    np.min(scalar_insga),
                    np.quantile(scalar_insga, 0.25),
                    np.quantile(scalar_insga, 0.75),
                    np.max(scalar_insga),
                    np.median(norm_insga),
                    np.min(norm_insga),
                    np.quantile(norm_insga, 0.25),
                    np.quantile(norm_insga, 0.75),
                    np.max(norm_insga),
                    int_nsga.population.objectives.shape[0],
                    int_nsga._function_evaluation_count,
                ]
                data_row[
                    ["NIMBUS-RVEA" + excess_col for excess_col in excess_columns]
                ] = [
                    np.median(scalar_nrvea),
                    np.min(scalar_nrvea),
                    np.quantile(scalar_nrvea, 0.25),
                    np.quantile(scalar_nrvea, 0.75),
                    np.max(scalar_nrvea),
                    np.median(norm_nrvea),
                    np.min(norm_nrvea),
                    np.quantile(norm_nrvea, 0.25),
                    np.quantile(norm_nrvea, 0.75),
                    np.max(norm_nrvea),
                    nimb_rvea.population.objectives.shape[0],
                    nimb_rvea._function_evaluation_count,
                ]
                data_row[
                    ["NIMBUS-NSGAIII" + excess_col for excess_col in excess_columns]
                ] = [
                    np.median(scalar_nnsga),
                    np.min(scalar_nnsga),
                    np.quantile(scalar_nnsga, 0.25),
                    np.quantile(scalar_nnsga, 0.75),
                    np.max(scalar_nnsga),
                    np.median(norm_nnsga),
                    np.min(norm_nnsga),
                    np.quantile(norm_nnsga, 0.25),
                    np.quantile(norm_nnsga, 0.75),
                    np.max(norm_nnsga),
                    nimb_nsga.population.objectives.shape[0],
                    nimb_nsga._function_evaluation_count,
                ]
                data = data.append(data_row, ignore_index=1)


data.to_csv("./results/data.csv", index=False)
