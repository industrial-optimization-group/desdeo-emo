from pyrvea.Problem.evonn_test_functions import EvoNNTestProblem
from pyrvea.Problem.evonn_problem import EvoNNModel

test_prob = EvoNNTestProblem("Three-hump camel")
training_data_input, training_data_output = test_prob.create_training_data(
    samples=150, method="random"
)

model_f1 = EvoNNModel(name="EvoNN_three_hump_camel")
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