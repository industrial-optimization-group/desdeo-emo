import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import desdeo

hypdata=pk.load(open("hypervolume.p", "rb"))
hyp = pd.DataFrame(hypdata)
plt.ion()
plt.rcParams.update({'font.size': 20})
figure = hyp.boxplot(showfliers=False)
plt.xlabel('Number of generations')
plt.ylabel('Normalized Hypervolume')
for i,d in enumerate(hyp):
    y = hyp[d]
    x = np.random.normal(i+1, 0.04, len(y))
    plt.scatter(x,y)

plt.show(block=True)
