from utils.misc import *


class Layer:

    def __init__(self, nodes, previous_layer=None, weights=None, bias=None, learning_rate=1):  # TODO  2021-02-28: Maybe a layer settings object?
        self.learning_rate = learning_rate
        self.previous_layer = previous_layer
        self.number_of_nodes = nodes
        self.next_layer = None
        self.activation = np.zeros([self.number_of_nodes, 1])

        if self.previous_layer:
            self.set_previous_layer(self.previous_layer)

        self.Z = 0
        self.dCost_dY = None

        self.weights = weights
        if self.weights is None:
            self.total_dCost_dW = 0
        else:
            self.total_dCost_dW = np.zeros(self.weights.shape)

        self.bias = bias
        if self.bias is None:
            self.total_dCost_dB = 0
        else:
            self.total_dCost_dB = np.zeros(self.bias.shape)

    def __len__(self):
        return self.number_of_nodes

    def activate(self):
        self.Z = np.dot(self.weights, self.previous_layer.activation) + self.bias
        self.activation = vectorised_sigmoid(self.Z)
        return self.activation

    def set_previous_layer(self, layer):
        self.previous_layer = layer
        self.previous_layer.next_layer = self

    def randomise_weights(self):
        self.weights = np.random.uniform(size=(self.number_of_nodes, len(self.previous_layer)),
                                         low=0.1,
                                         high=1)

    def randomise_bias(self):
        self.bias = np.random.uniform(size=(self.number_of_nodes, 1),
                                      low=0.1,
                                      high=1)

    def apply_accumulations(self):
        self.weights = self.weights - self.learning_rate * self.total_dCost_dW
        self.bias = self.bias - self.learning_rate * self.total_dCost_dB
        self.total_dCost_dW = np.zeros(self.weights.shape)
        self.total_dCost_dB = np.zeros(self.bias.shape)

    def accumulate_weights(self):
        self.total_dCost_dW += self._dCost_dW()

    def accumulate_bias(self):
        self.total_dCost_dB += self._dCost_dB()

    def cost(self, y):
        diff = y - self.activation
        diff_squared = diff ** 2
        total_diff_squared = diff_squared.sum()
        m = self.activation.size
        return total_diff_squared / (2 * m)

    def calculate_dCost_dY(self, y=None):
        if y is None:
            self.dCost_dY = np.dot(self.next_layer.dZ_dA(), self.next_layer.dCost_dZ())
        else:
            y_minus_y_hat = y - self.activation
            self.dCost_dY = - y_minus_y_hat / y.size
        return self.dCost_dY

    def dY_dZ(self):
        return self.activation * (1 - self.activation)

    def dZ_dW(self):
        return self.previous_layer.activation.T

    def dZ_dB(self):
        return np.ones((self.number_of_nodes, 1), dtype=float)

    def dZ_dA(self):
        return self.weights.T

    def dCost_dZ(self):
        return self.dCost_dY * self.dY_dZ()

    def _dCost_dW(self):
        return np.dot(self.dCost_dZ(), self.dZ_dW())

    def _dCost_dB(self):
        multiply = self.dCost_dZ() * self.dZ_dB()
        sum_of_rows = np.dot(multiply, np.ones(multiply.shape[1]))[np.newaxis].T
        return sum_of_rows
