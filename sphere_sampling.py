import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dist_based_translation(vectors):
    """Translates points towards origin based on distance."""
    dist = np.sqrt(np.sum(np.square(vectors), axis=1))
    max_dist = np.amax(dist)
    # max_dist = 1
    alpha = 2
    ratio = alpha * (1 / (dist * dist) - 1 / (max_dist * max_dist))
    t_factor = 1 / (1 + ratio)
    return vectors * t_factor[np.newaxis].T


def fibonacci(samples=1000, randomize=True):

    rnd = 1.0
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

    return points.append([z, x, y])


def random_sphere(N=600, dim=3):

    norm = np.random.normal
    normal_deviates = norm(size=(dim, N))

    radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
    points = normal_deviates / radius

    return points.T


fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)
plt.ion()
plt.show()
points = np.asarray(random_sphere())

points = np.delete(points, np.where(points[:, 2] < 0), axis=0)
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# plt.show(block=True)

newpoints = dist_based_translation(points[:, 0:-1])
z = np.sqrt(1 - np.sum(np.square(newpoints), axis=1))
ax.scatter(newpoints[:, 0], newpoints[:, 1], z)

#ax.set_aspect("equal")
plt.show(block=True)
