from pyrvea.Problem.evonn_test_functions import EvoNNTestProblem
from pyrvea.Problem.evonn_problem import EvoNNModel
from pyrvea.Problem.evodn2_problem import EvoDN2Model
from pyrvea.Problem.testProblem import testProblem
import numpy as np

# test_prob = EvoNNTestProblem("Three-hump camel")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=150, method="random"
# )

# test_prob = testProblem('ZDT2', 30, 2, 0, 1, 0)
# training_data_input = np.random.rand(150,30)
# training_data_output = np.asarray([test_prob.objectives(x) for x in training_data_input])

test_prob = EvoNNTestProblem("Sphere")
training_data_input, training_data_output = test_prob.create_training_data(
    samples=250, method="random"
)

model_evodn2 = EvoDN2Model(name="EvoDN2_" + test_prob.name)
model_evodn2.set_params(
        pop_size=500,
        subnets=(6, 10),
        num_nodes=10,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="min_error",
        logging=True,
        plotting=True)

model_evonn = EvoNNModel(name="EvoNN_" + test_prob.name)
model_evonn.set_params(
        num_nodes=20,
        pop_size=500,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="akaike_corrected",
        logging=True,
        plotting=True)

model_evodn2.fit(training_data_input, training_data_output)
model_evonn.fit(training_data_input, training_data_output)

# test_prob = EvoNNTestProblem("Matyas")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("Himmelblau")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("Rastigrin", num_of_variables=2)
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("Three-hump camel")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("Goldstein-Price")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("LeviN13")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# test_prob = EvoNNTestProblem("SchafferN2")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=250, method="random"
# )
#
# model_evodn2.fit(training_data_input, training_data_output)
# model_evonn.fit(training_data_input, training_data_output)
#
# # ZDT 1 & 2
test_prob = testProblem('ZDT6', 10, 2, 0, 1, 0)
training_data_input = np.random.rand(250, 10)
training_data_output = np.asarray([test_prob.objectives(x) for x in training_data_input])

f1_training_data_output = training_data_output[:, 0]
f2_training_data_output = training_data_output[:, 1]

model_evodn_f1 = EvoDN2Model(name="EvoDN2_" + test_prob.name + "_f1")
model_evodn_f1.set_params(
        pop_size=500,
        subnets=(6, 10),
        num_nodes=10,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="min_error",
        logging=True,
        plotting=True)

model_evodn_f2 = EvoDN2Model(name="EvoDN2_" + test_prob.name + "_f2")
model_evodn_f2.set_params(
        pop_size=500,
        subnets=(6, 10),
        num_nodes=10,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="min_error",
        logging=True,
        plotting=True)
model_evodn_f1.fit(training_data_input, f1_training_data_output)
model_evodn_f2.fit(training_data_input, f2_training_data_output)
#
# test_prob = testProblem('ZDT2', 30, 2, 0, 1, 0)
# training_data_input = np.random.rand(250, 30)
# training_data_output = np.asarray([test_prob.objectives(x) for x in training_data_input])
#
# f1_training_data_output = training_data_output[:, 0]
# f2_training_data_output = training_data_output[:, 1]
#
# model_evodn_f1.fit(training_data_input, f1_training_data_output)
# model_evodn_f2.fit(training_data_input, f2_training_data_output)
