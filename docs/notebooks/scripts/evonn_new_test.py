# %%
from desdeo_emo.surrogatemodelling.EvoNN import EvoNN
from desdeo_emo.surrogatemodelling.BioGP import BioGP
from desdeo_emo.surrogatemodelling.EvoDN2 import EvoDN2
import numpy as np
import pandas as pd
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII


# %%

X = np.random.rand(50, 3)
y = X[:, 0] * X[:, 1] + X[:, 2]
y = y.reshape(-1, 1)
data = pd.DataFrame(np.hstack((X, y)), columns=["x1", "x2", "x3", "y"])


# %% BioGP
"""model = BioGP(pop_size=100, training_algorithm=NSGAIII)
model.fit(data[["x1", "x2", "x3"]], data['y']) """


# %% EvoNN
"""model = EvoNN()
model.fit(data[["x1", "x2", "x3"]], data['y'])"""


# %% EvoDN2
"""model = EvoDN2(training_algorithm=RVEA)
model.fit(data[["x1", "x2", "x3"]], data['y'])"""


# %%
