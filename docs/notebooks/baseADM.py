import numpy as np
import plotly.graph_objects as go

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from pygmo import fast_non_dominated_sorting as nds


def visualize_2D_front_rvs(front, vectors: ReferenceVectors):
    fig = go.Figure(
        data=go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    for i in range(0, vectors.number_of_vectors):
        fig.add_trace(
            go.Scatter(
                x=[0, vectors.values[i, 0], vectors.values[i, 0]],
                y=[0, vectors.values[i, 1], vectors.values[i, 1]],
                name="vector #" + str(i + 1),
                marker=dict(size=1, opacity=0.8),
                line=dict(width=2),
            )
        )
    return fig


def visualize_3D_front_rvs(front, vectors: ReferenceVectors):

    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    for i in range(0, vectors.number_of_vectors):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vectors.values[i, 0], vectors.values[i, 0]],
                y=[0, vectors.values[i, 1], vectors.values[i, 1]],
                z=[0, vectors.values[i, 2], vectors.values[i, 2]],
                name="vector #" + str(i + 1),
                marker=dict(size=1, opacity=0.8),
                line=dict(width=2),
            )
        )
    return fig


def visualize_2D_front_rp(front, rp):
    fig = go.Figure(
        data=go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[rp[0]], y=[rp[1]], name="Reference point", mode="markers", marker_size=5,
        )
    )
    return fig


def visualize_3D_front_rp(front, rp):
    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[rp[0]],
            y=[rp[1]],
            z=[rp[2]],
            name="Reference point",
            mode="markers",
            marker_size=5,
        )
    )
    return fig


def generate_composite_front(*fronts):

    _fronts = np.vstack(fronts)

    cf = _fronts[nds(_fronts)[0][0]]

    return cf


def translate_front(front, ideal):
    translated_front = np.subtract(front, ideal)
    return translated_front


def normalize_front(front, translated_front):
    translated_norm = np.linalg.norm(translated_front, axis=1)
    translated_norm = np.repeat(translated_norm, len(translated_front[0, :])).reshape(
        len(front), len(front[0, :])
    )

    translated_norm[translated_norm == 0] = np.finfo(float).eps
    normalized_front = np.divide(translated_front, translated_norm)
    return normalized_front


def assign_vectors(front, vectors: ReferenceVectors):
    cosine = np.dot(front, np.transpose(vectors.values))
    if cosine[np.where(cosine > 1)].size:
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        cosine[np.where(cosine < 0)] = 0

    # theta = np.arccos(cosine) #check this theta later, if needed or not
    assigned_vectors = np.argmax(cosine, axis=1)

    return assigned_vectors


class baseADM:
    def __init__(
        self, composite_front, vectors: ReferenceVectors,
    ):

        self.composite_front = composite_front
        self.vectors = vectors
        self.ideal_point = composite_front.min(axis=0)
        self.translated_front = translate_front(self.composite_front, self.ideal_point)
        self.normalized_front = normalize_front(
            self.composite_front, self.translated_front
        )
        self.assigned_vectors = assign_vectors(self.normalized_front, self.vectors)
