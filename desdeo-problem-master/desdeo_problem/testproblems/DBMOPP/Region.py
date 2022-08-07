import numpy as np
from desdeo_problem.testproblems.DBMOPP.utilities import euclidean_distance
from matplotlib.patches import Circle
from numpy import matlib
from shapely.geometry import Point


class Attractor:
    """
    Attractor class.
    """

    def __init__(self) -> None:
        self._locations = None

    @property
    def locations(self):
        return self._locations

    @locations.setter
    def locations(self, value):
        self._locations = value

    def get_minimum_distance(self, x):
        d = euclidean_distance(self.locations, x)
        return np.min(d)


class Region:
    """
    Region class.
    """

    def __init__(self, centre: np.ndarray = None, radius: float = None):
        self._centre = centre
        self._radius = radius

    @property
    def centre(self):
        return self._centre

    @centre.setter
    def centre(self, value):
        self._centre = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    def is_close(self, x: np.ndarray, eps=1e-16):
        return self.radius + eps >= self.get_distance(x)

    def is_inside(self, x: np.ndarray, include_boundary=False):
        if include_boundary:
            return self.get_distance(x) <= self.radius
        return self.get_distance(x) < self.radius

    # x atleast 2d here too?
    def get_distance(self, x: np.ndarray):
        return euclidean_distance(self.centre, x)

    # TODO: check, might be incorrect
    def calc_location(self, a, rotation):
        radiis = matlib.repmat(self.radius, 1, 2)
        return self.centre + radiis * np.hstack((np.cos(a + rotation), np.sin(a + rotation)))

    # used to plot Region Centre for debugging purposes
    def plot(self, color, ax):
        x = self.centre[0]
        y = self.centre[1]
        circle = Circle((x, y), self.radius, fc=color, fill=True, alpha=0.5)
        ax.add_patch(circle)


class AttractorRegion(Region):
    """
    AttractorRegion implements region.
    """

    def __init__(self, locations, indices, centre, radius, convhull):
        self.locations = locations
        self.objective_indices = indices
        super().__init__(centre, radius)
        self.convhull = convhull

    def in_hull(self, x):
        shapeX = Point(x)
        if self.convhull is not None:
            return self.convhull.contains(shapeX)
        else:
            if self.locations.shape[0] == 1:
                return self.locations == x
            else:
                pass
                # check if between 2 points

    def plot(self, ax, color="b"):
        """
        Plots the attractorRegions
        """
        # if self.convhull is None: return

        # TODO: fix annotations
        n = self.locations.shape[0]
        p = np.atleast_2d(self.locations)

        # if not isinstance(self.convhull, ConvexHull):
        if n > 2:
            x, y = self.convhull.exterior.xy
            ax.plot(x, y, linewidth=1, color="black")
            ax.fill(x, y, color=color)
            # ax.scatter(x, y, s=5, color="blue")

            # annotate the objectives
            # for i in range(0,n):
            # ax.annotate(i, (float(x[i]), float(y[i]))) # annotating the points to draw the polygons for test
            #    ax.scatter(p[i,0], p[i,1], s=0.5,  color='black')
            #    ax.annotate(i, (p[i,0], p[i,1]))

        else:
            # for points
            if p.shape[0] == 1:
                for i in range(len(p)):
                    ax.scatter(self.locations[:, 0], self.locations[:, 1], color=color)
                    # ax.scatter(p[i, 0], p[i, 1], s=1, color="blue")
                    # ax.annotate(i, (p[i,0], p[i,1]))
            else:
                # for lines
                for i in range(len(p)):
                    ax.plot(self.locations[:, 0], self.locations[:, 1], color=color)
                    # ax.scatter(p[i, 0], p[i, 1], s=1, color="blue")
                    # ax.annotate(i, (p[i,0], p[i,1]))
