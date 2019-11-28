from desdeo_emo.surrogatemodelling.EvoNN import EvoNN
import numpy as np
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

X = np.random.rand(50, 3)
y = X[:, 0] * X[:, 1] + X[:, 2]
y = y.reshape(-1, 1)

model = EvoNN(training_algorithm=NSGAIII)
model.fit(X,y)