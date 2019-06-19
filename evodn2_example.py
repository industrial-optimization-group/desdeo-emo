from pyrvea.Problem.evonn_test_functions import EvoNNTestProblem
from pyrvea.Problem.evonn_problem import EvoNNModel
from pyrvea.Problem.evodn2_problem import EvoDN2Model

# test_prob = EvoNNTestProblem("Three-hump camel")
# training_data_input, training_data_output = test_prob.create_training_data(
#     samples=150, method="random"
# )

test_prob = EvoNNTestProblem("Sphere", num_of_variables=10)
training_data_input, training_data_output = test_prob.create_training_data(
    samples=150, method="random"
)

model_f1 = EvoDN2Model(name="EvoDN2_" + test_prob.name)
model_f1.set_params(
        num_hidden_nodes=15,
        pop_size=500,
        activation_func="sigmoid",
        opt_func="llsq",
        loss_func="rmse",
        criterion="akaike_corrected",
        logging=True,
        plotting=True)

model_f1.fit(training_data_input, training_data_output)
