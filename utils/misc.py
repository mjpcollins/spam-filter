import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


vectorised_sigmoid = np.vectorize(sigmoid)
