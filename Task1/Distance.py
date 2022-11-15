import numpy as np


def euclidean_distance(lhs, rhs):
    distance = np.sqrt(((lhs - rhs) ** 2).sum(axis=1))
    return np.array(distance)
