import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.MOEAD_NUMS import MOEA_D_NUMS
from desdeo_emo.EAs.MOEAD_NUMS_PLUS import MOEA_D_NUMS_PLUS
from pymoo.indicators.rmetric import RMetric

from pymoo.factory import get_problem, get_reference_directions
#import rmetric as rm
from sklearn.preprocessing import Normalizer
#from pymoo.configuration import Configuration

#Configuration.show_compile_hint = False

problem_names = ["DTLZ2"]
n_objs = np.asarray([3,5,7,9])  # number of objectives
K = 10
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = 50  # number of generations per iteration

algorithms = ["MOEA/D-NUMS", "MOEA/D-NUMS+"]  # algorithms to be compared

# the followings are for formatting results
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
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors


counter = 1
#total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for n_obj, n_var in zip(n_objs, n_vars):
    for problem_name in problem_names:
        print(f"Loop {problem_name} of {n_obj}")
        counter += 1
        problem = test_problem_builder(
            name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
        )
        
        problem.ideal = np.asarray([0] * n_obj)
        problem.nadir = np.asarray([1] * n_obj)
        true_nadir = np.asarray([1] * n_obj)
        # interactive
        int_nums = MOEA_D_NUMS(problem=problem, interact=True, n_gen_per_iter=num_gen_per_iter, roi_size=0.5)
        int_nums_plus = MOEA_D_NUMS_PLUS(problem=problem, interact=True, n_gen_per_iter=num_gen_per_iter, roi_size=0.5, num_solutions_display=5)
        int_nums.start()
        int_nums_plus.start()
        # initial reference point is specified randomly
        response = np.random.rand(n_obj)
        print(response)
        
        # run algorithms once with the randomly generated reference point
        pref_int_nums, plot_int_nums = int_nums.requests()
        pref_int_nums_plus, plot_int_nums_plus = int_nums_plus.requests()
        pref_int_nums[2].response = pd.DataFrame(
            [response], columns=pref_int_nums[2].content["dimensions_data"].columns
        )
        
        pref_int_nums_plus[2].response = pd.DataFrame(
            [response], columns=pref_int_nums_plus[2].content["dimensions_data"].columns
        )
        pref_int_nums, plot_int_nums = int_nums.iterate(pref_int_nums[2])
        pref_int_nums_plus, plot_int_nums_plus  = int_nums_plus.iterate(pref_int_nums_plus[2])
        # build initial composite front
        cf = generate_composite_front(
            int_nums.population.objectives, int_nums_plus.population.objectives
        )
        # the following two lines for getting pareto front by using pymoo framework
        problemR = get_problem(problem_name.lower(), n_var, n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        pareto_front = problemR.pareto_front(ref_dirs)
        # creates uniformly distributed reference vectors
        reference_vectors = ReferenceVectors(lattice_resolution, n_obj)
        # learning phase
        for i in range(L):
            data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                problem_name,
                n_obj,
                i + 1,
                num_gen_per_iter,
            ]
            # After this class call, solutions inside the composite front are assigned to reference vectors
            base = baseADM(cf, reference_vectors)
            # generates the next reference point for the next iteration in the learning phase
            response = gp.generateRP4learning(base)
            print("learning phase")
            print(response)
            data_row["reference_point"] = [
                response,
            ]
           
            # run algorithms with the new reference point
            pref_int_nums[2].response = pd.DataFrame(
                [response], columns=pref_int_nums[2].content["dimensions_data"].columns
            )
            pref_int_nums_plus[2].response = pd.DataFrame(
                [response], columns=pref_int_nums_plus[2].content["dimensions_data"].columns
            )
            pref_int_nums, plot_int_nums = int_nums.iterate(pref_int_nums[2])
            pref_int_nums_plus, plot_int_nums_plus = int_nums_plus.iterate(pref_int_nums_plus[2])
            # extend composite front with newly obtained solutions
            cf = generate_composite_front(
                cf, int_nums.population.objectives, int_nums_plus.population.objectives
            )
            # R-metric calculation
            ref_point = response.reshape(1, n_obj)
            # normalize reference point
            rp_transformer = Normalizer().fit(ref_point)
            norm_rp = rp_transformer.transform(ref_point)
            rmetric = RMetric(problemR, norm_rp, pf=pareto_front)
            # normalize solutions before sending r-metric
            NUMS_transformer = Normalizer().fit(int_nums.population.objectives)
            norm_NUMS = NUMS_transformer.transform(int_nums.population.objectives)
            NUMS_PLUS_transformer = Normalizer().fit(int_nums_plus.population.objectives)
            norm_NUMS_PLUS = NUMS_PLUS_transformer.transform(int_nums_plus.population.objectives)
            # R-metric calls for R_IGD and R_HV
            rigd_NUMS, rhv_NUMS = rmetric.do(norm_NUMS, others=norm_NUMS_PLUS)
            rigd_NUMS_PLUS, rhv_NUMS_PLUS = rmetric.do(norm_NUMS_PLUS, others=norm_NUMS)
            data_row[["NUMS" + excess_col for excess_col in excess_columns]] = [
                rigd_NUMS,
                rhv_NUMS,
            ]
            data_row[["NUMS_P" + excess_col for excess_col in excess_columns]] = [
                rigd_NUMS_PLUS,
                rhv_NUMS_PLUS,
            ]
            data = data.append(data_row, ignore_index=1)
        # Decision phase
        # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
        max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)
        for i in range(D):
            data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                problem_name,
                n_obj,
                L + i + 1,
                num_gen_per_iter,
            ]
            # since composite front grows after each iteration this call should be done for each iteration
            base = baseADM(cf, reference_vectors)
            # generates the next reference point for the decision phase
            response = gp.generateRP4decision(base, max_assigned_vector[0])
            data_row["reference_point"] = [
                response,
            ]
            # run algorithms with the new reference point
            pref_int_nums[2].response = pd.DataFrame(
                [response], columns=pref_int_nums[2].content["dimensions_data"].columns
            )
            pref_int_nums_plus[2].response = pd.DataFrame(
                [response], columns=pref_int_nums_plus[2].content["dimensions_data"].columns
            )
            pref_int_nums, plot_int_nums = int_nums.iterate(pref_int_nums[2])
            pref_int_nums_plus, plot_int_nums_plus = int_nums_plus.iterate(pref_int_nums_plus[2])
            # extend composite front with newly obtained solutions
            cf = generate_composite_front(
                cf, int_nums.population.objectives, int_nums_plus.population.objectives
            )
            # R-metric calculation
            ref_point = response.reshape(1, n_obj)
            rp_transformer = Normalizer().fit(ref_point)
            norm_rp = rp_transformer.transform(ref_point)
            # for decision phase, delta is specified as 0.2
            rmetric = RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)
            # normalize solutions before sending r-metric
            NUMS_transformer = Normalizer().fit(int_nums.population.objectives)
            norm_NUMS = NUMS_transformer.transform(int_nums.population.objectives)
            NUMS_PLUS_transformer = Normalizer().fit(int_nums_plus.population.objectives)
            norm_NUMS_PLUS = NUMS_PLUS_transformer.transform(int_nums_plus.population.objectives)
            rigd_NUMS, rhv_NUMS = rmetric.do(norm_NUMS, others=norm_NUMS_PLUS)
            rigd_NUMS_PLUS, rhv_NUMS_PLUS = rmetric.do(norm_NUMS_PLUS, others=norm_NUMS)
            data_row[["NUMS" + excess_col for excess_col in excess_columns]] = [
                rigd_NUMS,
                rhv_NUMS,
            ]
            data_row[["NUMS+" + excess_col for excess_col in excess_columns]] = [
                rigd_NUMS_PLUS,
                rhv_NUMS_PLUS,
            ]
            data = data.append(data_row, ignore_index=1)

data.to_csv("./results/output21.csv", index=False)
