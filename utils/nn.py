from utils.layer import Layer


class NeuralNetwork:

    def __init__(self, layer_settings):
        self.layers = [Layer(**settings) for settings in layer_settings]
        for indx, layer in enumerate(self.layers[1:]):
            layer.set_previous_layer(self.layers[indx])
        self.training_data = None
        self.training_labels = None

    def predict(self, input_array):
        self.layers[0].activation = input_array
        for layer in self.layers[1:]:
            layer.activate()
        return self.layers[-1].activation

    def set_training_data(self, dataset, labels):
        self.training_data = dataset
        self.training_labels = labels

    def train(self, epochs):
        for layer in self.layers[1:]:
            if layer.bias is None:
                layer.randomise_bias()
            if layer.weights is None:
                layer.randomise_weights()

        for _ in range(epochs):
            self.layers[0].activation = self.training_data
            for layer in self.layers[1:]:
                layer.activate()

            self.layers[2].calculate_dCost_dY(y=self.training_labels)
            for layer in self.layers[1:-1][::-1]:
                layer.calculate_dCost_dY()

            for layer in self.layers[1:][::-1]:
                layer.accumulate_weights()
                layer.accumulate_bias()
                layer.apply_accumulations()

