## Creating surrogate models in Python with EvoNN and EvoDN2

This example will show how to create surrogate models for multi-objective optimization problems using Evolutionary (Deep) Neural Networks. The code is currently capable of training the models with two algorithms, EvoNN and EvoDN2. EvoNN uses artificial, single layer neural networks, whereas EvoDN2 uses deep neural networks with multiple hidden layers and subnets.

The basic workflow is as follows:
1. Create a test problem with training data
2. Create a dataproblem class which handles the training of the models
3. After training, create a new population and evolve it using the models to optimize the problem

First, create the test problem:
```python
import numpy as np
import pandas as pd
from pyrvea.Problem.test_functions import OptTestFunctions
from pyrvea.Problem.dataproblem import DataProblem

# OptTestFunctions contains a number of testing functions.
# Schaffer function N.1, -100 <= x <= 100
# Minimize:
# f1 = x ** 2
# f2 = (x - 2) ** 2
test_prob = OptTestFunctions("SchafferN1", num_of_variables=1)

# Random data for training.
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
Set parameters for the Evolutionary Algorithm (as a dict). Check 
ea_parameters = {
    "target_pop_size": 50,
    "generations_per_iteration": 10,
    "iterations": 10,
}
Train the models.

```
problem.train(model_type="EvoNN", algorithm=PPGA, num_nodes=25, generations_per_iteration=10, iterations=10)
```
EvoNN and EvoDN2 models can currently be trained with Predator-Prey (PPGA) or reference vector guided evolutionary algorithms (RVEA). For explanations of the different EAs, see their respective class documentation at pyRVEA/EAs.
Training parameters can currently be passed as kwargs. For available parameters, see pyrvea.Problem.evonn_problem.EvoNNModel class documentation (a separate documentation page will come later). If no parameters are passed, defaults are used.

After the models have been trained, the test function can be optimized by creating a new population, passing the data problem class (containing the trained models) and calling evolve (PPGA or RVEA can be used for optimization):

```
pop = Population(problem)
pop.evolve(RVEA)
```
To show the final pareto plot:
```
pop.plot_pareto(filename="my-test-function")
```
