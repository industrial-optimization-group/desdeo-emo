import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
from pymoo.configuration import Configuration

Configuration.show_compile_hint = False


# problem_names = ["DTLZ2", "DTLZ3", "DTLZ4"]
problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
# n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])
n_objs = np.asarray([3])
K = 10
n_vars = K + n_objs - 1

num_gen_per_iter = [50]

algorithms = ["iRVEA", "iNSGAIII"]
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
)

excess_columns = [
    "_R_IGD",
    "_R_HV",
]

data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 0  # number of iterations for the learning phase
D = 20  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors

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

            # interactive
            int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
            int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)

            # initial reference point
            response = np.random.rand(n_obj)
            fig_rp = go.Figure()

            # run algorithms once with the randomly generated reference point
            _, pref_int_rvea = int_rvea.requests()
            _, pref_int_nsga = int_nsga.requests()
            pref_int_rvea.response = pd.DataFrame(
                [response], columns=pref_int_rvea.content["dimensions_data"].columns
            )
            pref_int_nsga.response = pd.DataFrame(
                [response], columns=pref_int_nsga.content["dimensions_data"].columns
            )

            _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
            _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

            cf = generate_composite_front(
                int_rvea.population.objectives, int_nsga.population.objectives
            )

            # the following two lines for getting pareto front by using pymoo framework
            problemR = get_problem(problem_name.lower(), n_var, n_obj)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            pareto_front = problemR.pareto_front(ref_dirs)

            # creates uniformly distributed reference vectors
            reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

            all_rps = np.empty(shape=(L + D, n_obj), dtype="object")

            for i in range(L):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]

                # problem_nameR = problem_name.lower()
                base = baseADM(cf, reference_vectors)

                response = gp.generateRP4learning(base)
                # print(response)

                data_row["reference_point"] = [
                    response,
                ]

                # Reference point generation for the next iteration
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_nsga.response = pd.DataFrame(
                    [response], columns=pref_int_nsga.content["dimensions_data"].columns
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)
                all_rps[i] = ref_point

                rmetric = rm.RMetric(problemR, norm_rp, pf=pareto_front)

                # normalize solutions before sending r-metric

                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                cf_transformer = Normalizer().fit(cf)
                norm_cf = cf_transformer.transform(cf)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                ]

                data = data.append(data_row, ignore_index=1)
                """fig = visualize_3D_front_rp(int_rvea.population.objectives, response)
                fig.write_html(
                    f"./results/decision_behaviour/iRVEA/"
                    f"iRVEA_{problem_name}_iteration_{i+1}.html"
                )
                fig = visualize_3D_front_rp(int_nsga.population.objectives, response)
                fig.write_html(
                    f"./results/decision_behaviour/iNSGA/"
                    f"iNSGA_{problem_name}_iteration_{i+1}.html"
                )"""

            # Decision phase
            base = baseADM(cf, reference_vectors)

            max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)
            # print(max_assigned_vector[0])

            for i in range(D):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    L + i + 1,
                    gen,
                ]

                # since composite front grows after each iteration this call should be done for each iteration
                base = baseADM(cf, reference_vectors)

                # generates the next reference point for the decision phase
                response = gp.generateRP4decision(base, max_assigned_vector[0])

                data_row["reference_point"] = [
                    response,
                ]

                # Reference point generation for the next iteration
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_nsga.response = pd.DataFrame(
                    [response], columns=pref_int_nsga.content["dimensions_data"].columns
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)
                all_rps[L + i] = ref_point

                # for decision phase, delta is specified as 0.2
                rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)

                # normalize solutions before sending r-metric

                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                cf_transformer = Normalizer().fit(cf)
                norm_cf = cf_transformer.transform(cf)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                ]

                data = data.append(data_row, ignore_index=1)
                """ fig = visualize_3D_front_rp(int_rvea.population.objectives, response)
                fig.write_html(
                    f"./results/decision_behaviour/iRVEA/"
                    f"iRVEA_{problem_name}_iteration_{L+i+1}.html"
                )
                fig = visualize_3D_front_rp(int_nsga.population.objectives, response)
                fig.write_html(
                    f"./results/decision_behaviour/iNSGA/"
                    f"iNSGA_{problem_name}_iteration_{L+i+1}.html"
                )"""
            fig_rp.add_trace(
                go.Scatter3d(
                    x=all_rps[:, 0],
                    y=all_rps[:, 1],
                    z=all_rps[:, 2],
                    name="Reference points",
                    mode="lines+markers",
                    marker_size=5,
                )
            )
            fig_rp.write_html(f"./results/decision/" f"RPs_{problem_name}_{gen}.html")
            fig = visualize_3D_front_rvs(cf, reference_vectors)
            fig.write_html(f"./results/decision/" f"cf_{problem_name}_{gen}.html")
            # print(all_rps)
data.to_csv("./results/decision/results_3objs_L0delta03_D20delta02.csv", index=False)

