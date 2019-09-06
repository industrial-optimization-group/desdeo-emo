from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline.offline import plot


def animate_init_(data: Union[np.ndarray, pd.DataFrame, list], filename: str) -> dict:
    """Plot the first (or zeroth) iteration of a population.

    Intended as a frames object. Plots Scatter for 2D and 3D data.
    Plots parallel coordinate plot for higher dimensional data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        Contains the data to be plotted. Each row is an individual's objective values.
    filename : str
        Contains the name of the file to which the plot is saved.

    Returns
    -------
    dict
        Plotly figure object
    """
    numobj = data.shape[1]
    if numobj == 2:
        figure = animate_2d_init_(data, filename)
    elif numobj == 3:
        figure = animate_3d_init_(data, filename)
    elif numobj >= 4:
        figure = animate_parallel_coords_init_(data, filename)
    print("Plot saved as: ", filename)
    print("View the plot by opening the file in browser.")
    print("To view the plot in Jupyter Notebook, use the IFrame command.")
    return figure


def animate_next_(
    data: Union[np.ndarray, pd.DataFrame, list],
    figure: dict,
    filename: str,
    generation: int = None,
) -> dict:
    """Plot the next set of individuals in an animation.

    Plots scatter for 2D and 3D data, parallel coordinate plot for 4D and up.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        The objective values to be plotted
    figure : dict
        Plotly figure object compatible dict
    filename : str
        Name of the file to which the plot is saved
    generation : int
        Iteration Number

    Returns
    -------
    dict
        Plotly Figure Object
    """

    numobj = data.shape[1]
    if generation is None:
        generation = len(figure["frames"])
    if numobj == 2:
        figure = animate_2d_next_(data, figure, filename, generation)
    elif numobj == 3:
        figure = animate_3d_next_(data, figure, filename, generation)
    elif numobj >= 4:
        figure = animate_parallel_coords_next_(data, figure, filename, generation)
    return figure


def animate_2d_init_(
    data: Union[np.ndarray, pd.DataFrame, list], filename: str
) -> dict:
    """Initiate a 2D scatter animation.

    Only for 2D data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        Objective values
    filename : str
        Name of the file to which plot is saved

    Returns
    -------
    dict
        Plotly Figure Object
    """

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


def animate_2d_next_(
    data: Union[np.ndarray, pd.DataFrame, list],
    figure: dict,
    filename: str,
    generation: int,
) -> dict:
    """Plot the next set of individuals in a 2D scatter animation.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        The objective values to be plotted
    figure : dict
        Plotly figure object compatible dict
    filename : str
        Name of the file to which the plot is saved
    generation : int
        Iteration Number

    Returns
    -------
    dict
        Plotly Figure Object
    """
    frame = {"data": [], "name": str(generation)}
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
            [generation],
            {
                "frame": {"duration": 300, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": generation,
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)
    return figure


def animate_3d_init_(
    data: Union[np.ndarray, pd.DataFrame, list], filename: str
) -> dict:
    """Plot the first (or zeroth) iteration of a population.

    Intended as a frames object. Plots Scatter 3D data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        Contains the data to be plotted. Each row is an individual's objective values.
    filename : str
        Contains the name of the file to which the plot is saved.

    Returns
    -------
    dict
        Plotly figure object
    """
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
    figure["layout"]["scene"] = dict(
        xaxis=dict(
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="rgb(255, 255, 255)",
            showbackground=True,
            zerolinecolor="rgb(255, 255, 255)",
        ),
        yaxis=dict(
            backgroundcolor="rgb(230, 200,230)",
            gridcolor="rgb(255, 255, 255)",
            showbackground=True,
            zerolinecolor="rgb(255, 255, 255)",
        ),
        zaxis=dict(
            backgroundcolor="rgb(230, 230,200)",
            gridcolor="rgb(255, 255, 255)",
            showbackground=True,
            zerolinecolor="rgb(255, 255, 255)",
        ),
    )

    data_dict = go.Scatter3d(
        x=list(data[:, 0]),
        y=list(data[:, 1]),
        z=list(data[:, 2]),
        mode="markers",
        marker=dict(
            size=8,
            color=data[:, 2],
            colorscale="Viridis",
            opacity=0.5,
            line=dict(width=2, color="black", colorscale="Viridis"),
        ),
    )
    figure["data"].append(data_dict)
    plot(figure, filename=filename)
    animate_3d_next_(data, figure, filename, 0)
    return figure


def animate_3d_next_(
    data: Union[np.ndarray, pd.DataFrame, list],
    figure: dict,
    filename: str,
    generation: int,
) -> dict:
    """Plot the next set of individuals in an animation.

    Plots scatter for 3D data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        The objective values to be plotted
    figure : dict
        Plotly figure object compatible dict
    filename : str
        Name of the file to which the plot is saved
    generation : int
        Iteration Number

    Returns
    -------
    dict
        Plotly Figure Object
    """
    frame = {"data": [], "name": str(generation)}
    sliders_dict = figure["layout"]["sliders"][0]
    data_dict = go.Scatter3d(
        x=list(data[:, 0]),
        y=list(data[:, 1]),
        z=list(data[:, 2]),
        mode="markers",
        marker=dict(
            size=8,
            color=data[:, 2],
            colorscale="Viridis",
            opacity=0.5,
            line=dict(width=2, color="black", colorscale="Viridis"),
        ),
    )
    frame["data"].append(data_dict)
    figure["frames"].append(frame)
    slider_step = {
        "args": [
            [generation],
            {
                "frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": generation,
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)
    return figure


def animate_parallel_coords_init_(
    data: Union[np.ndarray, pd.DataFrame, list], filename: str
) -> dict:
    """Plot the first (or zeroth) iteration of a population.

    Intended as a frames object. Plots parallel coordinate plot for >3D data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        Contains the data to be plotted. Each row is an individual's objective values.
    filename : str
        Contains the name of the file to which the plot is saved.

    Returns
    -------
    dict
        Plotly figure object
    """
    figure = {"data": [], "layout": {}, "frames": []}
    objectives = pd.DataFrame(data)
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
    dimensions = [
        dict(
            range=[min(objectives[column]), max(objectives[column])],
            label="f" + str(column),
            values=objectives[column],
        )
        for column in objectives
    ]
    line = dict(
        color=objectives[0],
        colorscale="Viridis",
        showscale=True,
        cmin=min(objectives[objectives.columns[0]]),
        cmax=max(objectives[objectives.columns[0]]),
    )
    data_dict = go.Parcoords(line=line, dimensions=dimensions)
    figure["data"].append(data_dict)
    plot(figure, filename=filename)
    animate_parallel_coords_next_(data, figure, filename, 0)
    return figure


def animate_parallel_coords_next_(
    data: Union[np.ndarray, pd.DataFrame, list],
    figure: dict,
    filename: str,
    generation: int,
) -> dict:
    """Plot the next set of individuals in an animation.

    Plots parallel coordinate plot for 4D and up.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, list]
        The objective values to be plotted
    figure : dict
        Plotly figure object compatible dict
    filename : str
        Name of the file to which the plot is saved
    generation : int
        Iteration Number

    Returns
    -------
    dict
        Plotly Figure Object
    """
    frame = {"data": [], "name": str(generation)}
    objectives = pd.DataFrame(data)
    sliders_dict = figure["layout"]["sliders"][0]
    dimensions = [
        dict(
            range=[min(objectives[column]), max(objectives[column])],
            label="f" + str(column),
            values=objectives[column],
        )
        for column in objectives
    ]
    line = dict(
        color=objectives[0],
        colorscale="Viridis",
        showscale=True,
        cmin=min(objectives[objectives.columns[0]]),
        cmax=max(objectives[objectives.columns[0]]),
    )
    data_dict = go.Parcoords(line=line, dimensions=dimensions)
    frame["data"].append(data_dict)
    figure["frames"].append(frame)
    slider_step = {
        "args": [
            [generation],
            {
                "frame": {"duration": 300, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": generation,
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)
    return figure


def test():
    data1 = np.random.rand(100, 3)
    data2 = np.square(data1)
    data3 = np.square(data2)
    data4 = np.square(data3)
    data = [data2, data3, data4]
    filename = "firsttest.html"
    figure = animate_3d_init_(data1, filename)
    for i in range(1, 4):
        figure = animate_3d_next_(data[i - 1], figure, filename, i)


def test2():
    data = np.random.rand(100, 1) * np.pi
    obj = np.hstack((data, np.sin(data), np.cos(data), np.log(data)))
    figure = animate_parallel_coords_init_(obj, "partest.html")
    for i in range(1, 10):
        obj = np.hstack((data, np.sin(data / i), np.cos(data / i), np.log(data / i)))
        figure = animate_parallel_coords_next_(obj, figure, "partest.html", i)
