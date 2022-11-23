import numpy as np


def euclidean_distance(lhs, rhs):
    return np.sum(((lhs - rhs) ** 2), -1) ** (1 / 2)
