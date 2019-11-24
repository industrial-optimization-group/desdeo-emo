from desdeo_emo.surrogatemodelling.EvoNN import EvoNN
import numpy as np

X = np.random.rand(50, 3)
y = X[:, 0] * X[:, 1] + X[:, 2]
y = y.reshape(-1, 1)

model = EvoNN()
model.fit(X,y)