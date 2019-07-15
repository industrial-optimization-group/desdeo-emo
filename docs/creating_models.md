## Creating surrogate models in Python with EvoNN and EvoDN2

This manual will guide the user to create surrogate models for multi-objective optimization problems using Evolutionary (Deep) Neural Networks. The code is currently capable of training and optimizing the models with two algorithms, EvoNN and EvoDN2. EvoNN uses artificial, single layer neural networks, whereas EvoDN2 uses deep neural networks with multiple hidden layers and subnets.

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
training_data_input, training_data_output = test_prob.create_training_data(
    samples=250, method="random"
)

# Convert numpy array to pandas dataframe
data = np.hstack((training_data_input, training_data_output))
dataset = pd.DataFrame.from_records(data)

# Set column names: x = variables, y = objectives. These are gotten automatically if importing .csv
x = []
for n in range(training_data_input.shape[1]):
    x.append("x" + str(n + 1))
y = ["f1", "f2"]
dataset.columns = x + y
```
After you have the data, create the DataProblem class and pass the data, variables and objectives.
x is a list of variable names, y is a list of objectives. The columns in the dataset should match these.
```
problem = DataProblem(data=dataset, x=x, y=y)
```
Split data into training and testing set:
```
problem.train_test_split(train_size=0.7)
```
Train the models.

```
problem.train(model_type="EvoNN", algorithm=PPGA, num_nodes=25, generations_per_iteration=10, iterations=10)
```
EvoNN and EvoDN2 models can currently be trained with Predator-Prey (PPGA) or reference vector guided evolutionary algorithms (RVEA). For explanations of the different EAs, see their respective class documentation at pyRVEA/EAs.
Training parameters can currently be passed as kwargs. For available parameters, see pyrvea.Problem.evonn_problem.EvoNNModel class documentation (a separate documentation page will come later). If no parameters are passed, defaults are used.

After the models have been trained, the test function can be optimized by creating a new population, passing the data problem class (containing the trained models) and calling evolve:

```
pop = Population(problem)
pop.evolve(RVEA)
```
To show the final pareto plot:
```
pop.plot_pareto(filename="my-test-function")
```
