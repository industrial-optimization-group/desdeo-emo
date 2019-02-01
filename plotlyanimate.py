from plotly.offline.offline import plot
import plotly.graph_objs as go
import numpy as np


def animate_2d_init_(data, filename):
    figure = {"data": [], "layout": {}, "frames": []}
    figure["layout"]["xaxis"] = {"autorange": True}
    figure["layout"]["yaxis"] = {"autorange": True}
    figure["layout"]["hovermode"] = "closest"
    figure["layout"]["sliders"] = {
        "args": ["transition", {"duration": 400, "easing": "cubic-in-out"}],
        "initialValue": "1952",
        "plotlycommand": "animate",
        "visible": True,
    }
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
    figure["layout"]["sliders"] = [sliders_dict]
    data_dict = {
        "x": list(data[:, 0]),
        "y": list(data[:, 1]),
        "mode": "markers",
        "marker": {
            "size": 5,
            "color": "rgba(255, 182, 193, .9)",
            "line": dict(width=2),
        },
    }
    figure["data"].append(data_dict)
    plot(figure, filename=filename)
    animate_2d_next_(data, figure, filename, 0)
    return figure


def animate_2d_next_(data, figure, filename, generation):
    frame = {"data": [], "name": str(generation + 1)}
    sliders_dict = figure["layout"]["sliders"][0]
    data_dict = {
        "x": list(data[:, 0]),
        "y": list(data[:, 1]),
        "mode": "markers",
        "marker": {
            "size": 5,
            "color": "rgba(255, 182, 193, .9)",
            "line": dict(width=2),
        },
    }
    frame["data"].append(data_dict)
    figure["frames"].append(frame)
    slider_step = {
        "args": [
            [generation + 1],
            {
                "frame": {"duration": 300, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": generation + 1,
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)
    return figure


def animate_3d_init_(data, filename):
    figure = {"data": [], "layout": {}, "frames": []}
    figure["layout"]["hovermode"] = "closest"
    figure["layout"]["sliders"] = {
        "args": ["transition", {"duration": 400, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "animate",
        "visible": True,
    }
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
    figure["layout"]["sliders"] = [sliders_dict]
    data_dict = go.Scatter3d(
        x=list(data[:, 0]),
        y=list(data[:, 1]),
        z=list(data[:, 2]),
        mode="markers",
        marker={"size": 5, "color": "rgba(255, 182, 193, .9)", "line": dict(width=2)},
    )
    figure["data"].append(data_dict)
    plot(figure, filename=filename)
    animate_3d_next_(data, figure, filename, 0)
    return figure


def animate_3d_next_(data, figure, filename, generation):
    frame = {"data": [], "name": str(generation + 1)}
    sliders_dict = figure["layout"]["sliders"][0]
    data_dict = go.Scatter3d(
        x=list(data[:, 0]),
        y=list(data[:, 1]),
        z=list(data[:, 2]),
        mode="markers",
        marker={"size": 5, "color": "rgba(255, 182, 193, .9)", "line": dict(width=2)},
    )
    frame["data"].append(data_dict)
    figure["frames"].append(frame)
    slider_step = {
        "args": [
            [generation + 1],
            {
                "frame": {"duration": 300, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": generation + 1,
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)
    return figure


data1 = np.random.rand(100, 3)
data2 = np.square(data1)
data3 = np.square(data2)
data4 = np.square(data3)
data = [data2, data3, data4]
filename = "firsttest.html"
figure = animate_3d_init_(data1, filename)
for i in range(1, 4):
    figure = animate_3d_next_(data[i - 1], figure, filename, i)
