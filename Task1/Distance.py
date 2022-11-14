import numpy as np


def euclidean_distance(lhs, rhs):
    return np.sqrt(((lhs - rhs) ** 2).sum(axis=1))
