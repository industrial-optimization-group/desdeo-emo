## Creating surrogate models in Python with EvoNN and EvoDN2

This example will show how to create surrogate models for multi-objective optimization problems using Evolutionary (Deep) Neural Networks. The code is currently capable of training the models with two algorithms, EvoNN and EvoDN2. EvoNN uses artificial, single layer neural networks, whereas EvoDN2 uses deep neural networks with multiple hidden layers and subnets.

The basic workflow is as follows:
1. Create a test problem with training data
2. Create a dataproblem class which handles the training of the models
3. After training, create a new population and evolve it using the models to optimize the problem

First, create the test problem:
```python
import numpy as np
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.Problem.dataproblem import DataProblem

# TestProblem class contains a number of testing functions
# Schaffer function N.1, -100 <= x <= 100
# Minimize:
# f1 = x ** 2
# f2 = (x - 2) ** 2

test_prob = TestProblem(name="SchafferN1")

# Random data for training
# x = list of variable names, y = list of objectives
dataset, x, y = test_prob.create_training_data(samples=500, method="random")

```
After you have the data, create the DataProblem class and pass the data, variables and objectives.
```
problem = DataProblem(data=dataset, x=x, y=y)
```
Split data into training and testing set:
```
problem.train_test_split(train_size=0.7)
```
Select the Evolutionary Algorithm you want to use and set the parameters (or use defaults). Check [documentation](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html) for details.

Train the models. Model specific parameters can be passed as kwargs. If no parameters are passed, defaults are used. See docs for [EvoNN parameters](https://htmlpreview.github.io/?https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/_build/html/pyrvea.Problem.html#pyrvea.Problem.evonn_problem.EvoNNModel.set_params) and [EvoDN2 parameters](https://htmlpreview.github.io/?https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/_build/html/pyrvea.Problem.html#pyrvea.Problem.evodn2_problem.EvoDN2Model.set_params).

```
from pyrvea.EAs.PPGA import PPGA

ea_params = {
    "target_pop_size": 100,
    "generations_per_iteration": 10,
    "iterations": 10,
}

problem.train(model_type="EvoNN", algorithm=PPGA, num_nodes=25, ea_parameters=ea_params)
```
EvoNN and EvoDN2 models can currently be trained with available algorithms in [pyRVEA/EAs](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html). For explanations of the different EAs, see their respective [documentation](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html).

After the models have been trained, the test problem can be optimized by creating a new population, passing the data problem class (containing the trained models) and calling evolve. EA parameters can be modified for optimization phase if wanted.

```
from pyrvea.EAs.RVEA import RVEA

pop = Population(problem)
opt_params = {"iterations": 10, "generations_per_iteration": 25}
pop.evolve(EA=RVEA, ea_parameters=opt_params)
```
To show the final pareto plot:
```
pop.plot_pareto(filename="my-test-function")
```
