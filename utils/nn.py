from utils.layer import Layer


class NeuralNetwork:

    def __init__(self, layer_setup):
        self.layers = [Layer(nodes=node_size) for node_size in layer_setup]
        for indx, layer in enumerate(self.layers[1:]):
            layer.set_previous_layer(self.layers[indx])
            layer.randomise_weights()
            layer.randomise_bias()

    def predict(self, input_array):
        activation = input_array
        for layer in self.layers[1:]:
            activation = layer.activate(previous_layer_activation=activation)
        return activation


