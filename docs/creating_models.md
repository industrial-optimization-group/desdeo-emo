## Creating surrogate models in Python

This manual will guide the user to create surrogate models for multi-objective optimization problems using Evolutionary (Deep) Neural Networks. The code is currently capable of training and optimizing the models with two algorithms, EvoNN and EvoDN2. EvoNN uses artificial, single layer neural networks, whereas EvoDN2 uses deep neural networks with multiple hidden layers.

For demonstration, Zitzler-Deb-Thiele’s (ZDT) test problems will be used.  
First, create the test problem. In this case, random data will be used, but data can easily be imported in .csv format as well with Pandas: 

```python
import pandas as pd
data = pd.read_csv("name_of_data_file.csv")
```
Create the test problem:
```python
import numpy as np
import pandas as pd
from pyrvea.Problem.testProblem import testProblem
from pyrvea.Problem.dataproblem import DataProblem

test_prob = testProblem(
    name="ZDT1",
    num_of_variables=30,
    num_of_objectives=2,
    num_of_constraints=0,
    upper_limits=1,
    lower_limits=0,
)

# Generate random data for training
training_data_input = np.random.rand(250, 30)
training_data_output = np.asarray([test_prob.objectives(x) for x in training_data_input])
data = np.hstack((training_data_input, training_data_output))

# Convert numpy array to pandas dataframe
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

