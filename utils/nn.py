import json
import numpy as np
from utils.layer import Layer


class NeuralNetwork:

    def __init__(self, layer_settings):
        self.layers = [Layer(**settings) for settings in layer_settings]
        for indx, layer in enumerate(self.layers[1:]):
            layer.set_previous_layer(self.layers[indx])
        self._training_data = None
        self._training_labels = None

    def predict(self, input_array):
        self.layers[0].activation = input_array.T
        for layer in self.layers[1:]:
            layer.activate()
        return self.layers[-1].activation.round(0)

    def classification_accuracy(self, dataset, labels):
        prediction = self.predict(dataset)
        squared_diff = (labels - prediction) ** 2
        total_squared_diff = squared_diff.sum()
        n = prediction.size
        return 1 - (total_squared_diff / n)

    def set_training_data(self, dataset, labels):
        self._training_data = dataset
        self._training_labels = labels

    def train(self, epochs, epoch_size=1):
        for layer in self.layers[1:]:
            if layer.bias is None:
                layer.randomise_bias()
            if layer.weights is None:
                layer.randomise_weights()

        for epoch_number in range(epochs):
            rows_in_each_segment = int(epoch_size * self._training_data.shape[1])
            for batch in range(int(1 / epoch_size)):
                training_dataset_batch = self._training_data[:, batch * rows_in_each_segment: (batch + 1) * rows_in_each_segment]
                training_labels_batch = self._training_labels[batch * rows_in_each_segment: (batch + 1) * rows_in_each_segment]
                accuracy = self.classification_accuracy(dataset=training_dataset_batch,
                                                        labels=training_labels_batch)

                self.layers[-1].calculate_dCost_dY(y=training_labels_batch)
                for layer in self.layers[1:-1][::-1]:
                    layer.calculate_dCost_dY()

                for layer in self.layers[1:][::-1]:
                    layer.accumulate_weights()
                    layer.accumulate_bias()
                    layer.apply_accumulations()

            if epoch_number % 1000 == 0:
                print(accuracy)
        print(accuracy)

    def load_weights_and_biases(self, filename):
        with open(filename, 'r') as F:
            weights_and_bias_data = json.load(F)
        for data in weights_and_bias_data:
            self.layers[data['layer']].weights = np.array(data['weights'])
            self.layers[data['layer']].bias = np.array(data['bias'])

    def save_weights_and_biases(self, filename):
        weights_and_bias_data = list()
        for layer_indx, layer in enumerate(self.layers):
            if layer.weights is None:
                weights = [[]]
            else:
                weights = layer.weights.tolist()
            if layer.bias is None:
                bias = [[]]
            else:
                bias = layer.bias.tolist()
            layer_data = {'layer': layer_indx,
                          'weights': weights,
                          'bias': bias}
            weights_and_bias_data.append(layer_data)
        with open(filename, 'w') as F:
            json.dump(obj=weights_and_bias_data,
                      fp=F)
