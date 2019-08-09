## Creating surrogate models in Python with EvoNN, EvoDN2 and BioGP

This example will show how to use the code and structure in pyRVEA to create models using EvoNN, EvoDN2 and BioGP ([see descriptions of the algorithms here](https://github.com/delamorte/pyRVEA/blob/master/docs/README.md)). The code is currently capable of training and optimizing the models with all of the genetic algorithms implemented in [pyRVEA/EAs](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html).

The basic workflow is as follows:
1. Create or import training data
2. Create a problem class which handles the training of the models
3. After training, create a new population and evolve it using the models to optimize the problem

For training data, example functions can be found in pyRVEA/Problem/testproblem.py and test_functions.py packages, or a custom data set can be imported.

Using [Fonseca-Fleming](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization) two objective function with 2 variables as an example:
```python
import numpy as np
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.Problem.dataproblem import DataProblem

test_prob = TestProblem(name="Fonseca-Fleming", num_of_variables=2)

# Random data for training
# x = list of variable names, y = list of objectives
dataset, x, y = test_prob.create_training_data(samples=500)
```
Or you can import a data set, here an example data set for [ZDT1](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization) problem with 30 variables and 1000 samples is used:
```
dataset = pd.read_excel("ZDT1_1000.xls", header=0)
x = dataset.columns[0:30].tolist()
y = dataset.columns[30:].tolist()
```
After you have the data, create the DataProblem class and pass the data, variables and objectives.
```
problem = DataProblem(data=dataset, x=x, y=y)
```
You can split the data into a training and testing set:
```
problem.train_test_split(train_size=0.7)
```
Select the genetic algorithm you want to use and set the parameters (or use defaults). Check [documentation](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html) for details.

Train the models. Model specific parameters can be passed as kwargs. If no parameters are passed, defaults are used. See docs for [EvoNN parameters](https://htmlpreview.github.io/?https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/_build/html/pyrvea.Problem.html#pyrvea.Problem.evonn_problem.EvoNNModel.set_params),  [EvoDN2 parameters](https://htmlpreview.github.io/?https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/_build/html/pyrvea.Problem.html#pyrvea.Problem.evodn2_problem.EvoDN2Model.set_params) and [BioGP parameters](https://htmlpreview.github.io/?https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/_build/html/pyrvea.Problem.html#pyrvea.Problem.biogp_problem.BioGPModel.set_params).

Both EA parameters and the model parameters can greatly affect the performance of the model. The best options depend on the problem, so experimenting with different values is encouraged.

```
from pyrvea.EAs.PPGA import PPGA

ea_params = {
    "generations_per_iteration": 10,
    "iterations": 10,
}

problem.train(model_type="EvoDN2", algorithm=PPGA, ea_parameters=ea_params)
```

Note that for BioGP, function set and terminal should be adjusted according to the problem. By default, function set includes addition, substraction, multiplication and division, and terminal set includes the variables from the data.
For [Fonseca-Fleming function](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization), square root and negative should be added to the function set and constants 1 and 2 to the terminal set:
```
f_set = ("add", "sub", "mul", "div", "sqrt", "neg")
t_set = [1, 2]
problem.train(
    model_type="BioGP",
    algorithm=RVEA,
    terminal_set=t_set,
    function_set=f_set,
)
```

The models can currently be trained with available algorithms in [pyRVEA/EAs](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html). For explanations of the different EAs, see their respective [documentation](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html).

The model's prediction vs. target values can be plotted as follows:
```
# Prediction for f1
f1_y_pred = problem.models["f1"][0].predict(problem.data[problem.x])

problem.models["f1"][0].plot(
    f1_y_pred, np.asarray(problem.data["f1"]), name="filename" + "f1"
)
# Prediction for f2
f2_y_pred = problem.models["f2"][0].predict(problem.data[problem.x])

problem.models["f2"][0].plot(
    f2_y_pred, np.asarray(problem.data["f2"]), name="filename" + "f2"
)
```

After the models have been trained, the objectives can be optimized by creating a new population, passing the data problem class (containing the trained models) and calling evolve. EA parameters can be modified for optimization phase if wanted.

```
from pyrvea.EAs.RVEA import RVEA

pop = Population(problem)
opt_params = {"iterations": 10, "generations_per_iteration": 25}
pop.evolve(EA=RVEA, ea_parameters=opt_params)
```
To show the final pareto plot:
```
pop.plot_pareto(name="my-test-function")
```
