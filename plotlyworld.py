import plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

py.offline.plot({
     "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
     "layout": go.Layout(title="hello world")
 }, auto_open=False)