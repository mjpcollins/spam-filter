import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def cost_function(prediction, label):
    return (prediction - label) ** 2


def cost_function_prime(label, prediction):
    return 2 * (label - prediction)  # TODO: 2021-03-04 - is this right...?


vectorised_sigmoid = np.vectorize(sigmoid)
vectorised_sigmoid_prime = np.vectorize(sigmoid_prime)
vectorised_cost_function = np.vectorize(cost_function)
vectorised_cost_function_prime = np.vectorize(cost_function_prime)
