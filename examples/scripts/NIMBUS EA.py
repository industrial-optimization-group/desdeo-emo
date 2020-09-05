import numpy as np
import pandas as pd

from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs.NIMBUS_EA import NIMBUS_RVEA


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


problem = test_problem_builder(name="ZDT1")


# In[5]:


problem.ideal = np.asarray([0, 0])
problem.nadir = np.asarray([1, 1])


# In[6]:


evolver = NIMBUS_RVEA(problem, n_gen_per_iter=700)


# In[7]:


plot, pref = evolver.requests()


# In[8]:


response = np.asarray([[0.1,0.9]])
pref.response = pd.DataFrame(response, columns=pref.content['dimensions_data'].columns)


# In[9]:


plot, pref = evolver.iterate(pref)


# In[12]:


nimbus = plt.scatter(x=evolver.population.objectives[:, 0], y=evolver.population.objectives[:, 1], label='NIMBUS-RVEA')
refp = plt.scatter(x=response[:,0], y=response[:,1], c='r', label='Reference Point')
stom = plt.arrow(x=response[0,0], y=response[0,1], dx=response[0,0]*10, dy=response[0,1]*10, color='g', label='STOM')
asf = plt.arrow(x=response[0,0], y=response[0,1], dx=10, dy=10,color='black',label="ASF")
plt.arrow(x=response[0,0], y=response[0,1], dx=-response[0,0]*10, dy=-response[0,1]*10, color='g')
plt.arrow(x=response[0,0], y=response[0,1], dx=-10, dy=-10, color='black')
pareto = plt.plot(zdt1_pareto[:,0], zdt1_pareto[:,1],'-.')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("ZDT1 problem with NIMBUS inspired EA")
plt.legend([nimbus, refp, stom, asf], ["NIMBUS-RVEA","Reference Point","STOM", "ASF"])
plt.gca().set_aspect('equal', adjustable='box')


# In[56]:


scaled_data = np.asarray([scalar(evolver.population.fitness, pref.response.values) for scalar in evolver.scalarization_methods]).T
# In[57]:


plt.scatter(x=scaled_data[:,0], y=scaled_data[:,1])


# In[29]:


evolver.population.objectives.shape


# In[11]:


zdt1_pareto = np.asarray([[0.0000000333,0.9998173820],
[0.0036524605,0.9395644103],
[0.0109353082,0.8954279758],
[0.0201661797,0.8579923253],
[0.0307984775,0.8245050499],
[0.0425521730,0.7937182195],
[0.0552536936,0.7649389578],
[0.0687834600,0.7377339899],
[0.0830533346,0.7118102455],
[0.0979952414,0.6869580836],
[0.1135547782,0.6630211013],
[0.1296872770,0.6398788023],
[0.1463553419,0.6174358329],
[0.1635270897,0.5956151713],
[0.1811749649,0.5743534743],
[0.1992748910,0.5535978371],
[0.2178056136,0.5333035102],
[0.2367481746,0.5134322508],
[0.2560856025,0.4939509881],
[0.2758025276,0.4748309533],
[0.2958850236,0.4560468554],
[0.3163203619,0.4375763502],
[0.3370968359,0.4193995902],
[0.3582036897,0.4014987972],
[0.3796309819,0.3838579856],
[0.4013694644,0.3664627364],
[0.4234104957,0.3492999957],
[0.4457460837,0.3323578176],
[0.4683687091,0.3156253153],
[0.4912712857,0.2990925270],
[0.5144471844,0.2827502636],
[0.5378901526,0.2665900514],
[0.5615942516,0.2506040755],
[0.5855539020,0.2347850615],
[0.6097637681,0.2191262791],
[0.6342188315,0.2036214270],
[0.6589142602,0.1882646612],
[0.6838455122,0.1730504779],
[0.7090082355,0.1579737323],
[0.7343982546,0.1430296070],
[0.7600116005,0.1282135580],
[0.7858444564,0.1135213164],
[0.8118931476,0.0989488651],
[0.8381541846,0.0844923897],
[0.8646242088,0.0701482867],
[0.8912999852,0.0559131474],
[0.9181784008,0.0417837401],
[0.9452564510,0.0277570000],
[0.9725312450,0.0138300121],
[1.0000000000,0.0000000000],])


# In[13]:


# 3 Objectives


# In[22]:


import plotly.graph_objects as go


# In[14]:


problem = test_problem_builder(name="DTLZ1", n_of_objectives=3, n_of_variables=5)


# In[15]:


problem.ideal = np.asarray([0, 0, 0])
problem.nadir = np.asarray([0.5, 0.5, 0.5])


# In[16]:


evolver = NIMBUS_RVEA(problem, n_gen_per_iter=700)


# In[17]:


plot, pref = evolver.requests()


# In[19]:


pref.content


# In[65]:


response = np.asarray([[0.45,0.1, 0.4]])
pref.response = pd.DataFrame(response, columns=pref.content['dimensions_data'].columns)


# In[66]:


plot, pref = evolver.iterate(pref)


# In[73]:


fig = plot3d(plot, response)
fig


# In[72]:


def plot3d(plot_file, response):
    # NIMBUSEA
    fig = go.Figure(
        go.Scatter3d(
            x=plot.content['data'][('f1',)],
            y=plot.content['data'][('f2',)],
            z=plot.content['data'][('f3',)],
            name="NIMBUS-RVEA",
            mode='markers',
            marker=dict(
            size=4,
            opacity=0.8
        )
        )
    )

    #True Front
    fig.add_trace(
        go.Scatter3d(
            x=[0.5, 0, 0, 0.5],
            y=[0, 0.5, 0, 0],
            z=[0, 0, 0.5, 0],
            name="True Front"
        )
    )

    #Nadir Point
    fig.add_trace(
        go.Scatter3d(
            x=[0.5],
            y=[0.5],
            z=[0.5],
            name="Nadir Point"
        )
    )



    # Reference Point

    fig.add_trace(
        go.Scatter3d(
            x=[response[0,0]],
            y=[response[0,1]],
            z=[response[0,2]],
            name="Reference Point"
        )
    )

    # STOM

    fig.add_trace(
        go.Scatter3d(
            x=[0, response[0,0]],
            y=[0, response[0,1]],
            z=[0, response[0,2]],
            name="STOM"
        )
    )

    #ASF

    fig.add_trace(
        go.Scatter3d(
            x=[response[0,0]-1, response[0,0]+1],
            y=[response[0,1]-1, response[0,1]+1],
            z=[response[0,2]-1, response[0,2]+1],
            name="ASF"
        )
    )

    # Ideal Point
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            name="Ideal Point"
        )
    )

    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-0.1,0.6],),
            yaxis = dict(range=[-0.1,0.6],),
            zaxis = dict(range=[-0.1,0.6],),),
                     )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
    


# In[ ]:


prob_dtlz2 = test_problem_builder(name="DTLZ2", n_of_objectives=3, n_of_variables=5)
problem.ideal = np.asarray([0, 0, 0])
problem.nadir = np.asarray([1, 1, 1])

