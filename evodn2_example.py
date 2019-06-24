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

test_prob = EvoNNTestProblem("Goldstein-Price")
training_data_input, training_data_output = test_prob.create_training_data(
    samples=250, method="random"
)

model_evodn = EvoDN2Model(name="EvoDN2_" + test_prob.name)
model_evodn.set_params(
        pop_size=500,
        subnets=(4, 8),
        num_nodes=10,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="min_error",
        logging=True,
        plotting=True)


model_evodn.fit(training_data_input, training_data_output)

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

model_evonn.fit(training_data_input, training_data_output)
