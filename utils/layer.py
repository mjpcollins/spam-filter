from utils.misc import *


class Layer:

    def __init__(self, nodes, previous_layer=None, weights=None, bias=None):  # TODO  2021-02-28: Maybe a layer settings object?
        self.previous_layer = previous_layer
        self.number_of_nodes = nodes
        self.next_layer = None
        self.activation = np.zeros([self.number_of_nodes, 1])

        if self.previous_layer:
            self.set_previous_layer(self.previous_layer)
        self.weights = weights
        self.bias = bias

        self.dC_dA = 0
        self.dZ_dA_minus_one = 0
        self.dA_dZ = 0
        self.Z = 0

    def __len__(self):
        return self.number_of_nodes

    def activate(self, previous_layer_activation=None):
        if previous_layer_activation is None:
            self.Z = np.dot(self.weights, self.previous_layer.activation) + self.bias
        else:
            self.Z = np.dot(self.weights, previous_layer_activation) + self.bias
        self.activation = vectorised_sigmoid(self.Z)
        return self.activation

    def set_previous_layer(self, layer):
        self.previous_layer = layer
        self.previous_layer.next_layer = self
        self.weights = np.zeros([self.number_of_nodes,
                                 len(self.previous_layer)])
        self.bias = np.zeros([self.number_of_nodes, 1])

    def randomise_weights(self):
        self.weights = np.random.rand(self.number_of_nodes,
                                      len(self.previous_layer))

    def randomise_bias(self):
        self.bias = np.random.rand(self.number_of_nodes, 1)

    def calculate_dZ_dA_minus_one(self):
        self.dZ_dA_minus_one = self.next_layer.weights
        return self.dZ_dA_minus_one

    def calculate_dA_dZ(self):
        self.dA_dZ = vectorised_sigmoid_prime(self.Z)
        return self.dA_dZ

    def calculate_dC_dA(self, y=None):
        if y is None:
            self.dC_dA = sum(np.dot(np.diag(self.next_layer.dA_dZ), self.dZ_dA_minus_one) * self.next_layer.dC_dA)
        else:
            self.dC_dA = vectorised_cost_function_prime(self.activation, y)

    def update_all_deltas(self):
        self.calculate_dZ_dA_minus_one()
        self.calculate_dA_dZ()
