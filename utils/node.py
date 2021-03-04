import numpy as np
from utils.misc import vectorised_sigmoid


x = np.array([[0, 1, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])


def get_next_layer_activation(weights, activation, bias):
    return vectorised_sigmoid(np.dot(weights, activation) + bias)


hidden_layer_nodes = 2
previous_layer_activation = np.random.rand(3, 1)
hidden_layer_weights = np.random.rand(hidden_layer_nodes, 3)  # n x m matrix (each row is the weights to the next layer neurons)
hidden_layer_bias = np.random.rand(hidden_layer_nodes, 1)  # m vector
next_layer_activation = get_next_layer_activation(weights=hidden_layer_weights,
                                                  activation=previous_layer_activation,
                                                  bias=hidden_layer_bias)

output_layer_nodes = 1
previous_layer_activation = np.random.rand(len(next_layer_activation), 1)
hidden_layer_weights = np.random.rand(output_layer_nodes, len(next_layer_activation))  # n x m matrix (each row is the weights to the next layer neurons)
hidden_layer_bias = np.random.rand(output_layer_nodes, 1)  # m vector
next_layer_activation = get_next_layer_activation(weights=hidden_layer_weights,
                                                  activation=previous_layer_activation,
                                                  bias=hidden_layer_bias)

y = np.array([0, 1, 1, 0])

