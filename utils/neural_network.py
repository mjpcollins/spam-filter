import json
import numpy as np
from utils.layer import Layer


class NeuralNetwork:

    def __init__(self, layer_settings):
        self.layers = [Layer(**settings) for settings in layer_settings]
        self._link_layers()
        self._training_features = None
        self._training_labels = None

    def predict(self, input_array):
        self.layers[0].activation = input_array.T
        for layer in self.layers[1:]:
            layer.activate()
        return self.layers[-1].activation.round(0)

    def classification_accuracy(self, features, labels):
        prediction = self.predict(features)
        squared_diff = (labels - prediction) ** 2
        total_squared_diff = squared_diff.sum()
        n = prediction.size
        return 1 - (total_squared_diff / n)

    def set_training_data(self, features, labels):
        self._training_features = features
        self._training_labels = labels

    def train(self, epochs, batch_size=100):
        self.ensure_layers_have_weights_and_bias()
        for epoch_number in range(epochs):
            number_of_batches = self._training_features.shape[1] // batch_size + 1
            for batch in range(number_of_batches):
                features, labels = self.get_training_batch(batch, batch_size)
                accuracy = self.forward_propagation(features, labels)
                self.backward_propagation(labels)
                if (epoch_number % 1000 == 0) and (batch == 0):
                    print(accuracy)

    def get_training_batch(self, batch, rows_in_segment):
        segment_from = batch * rows_in_segment
        segment_to = (batch + 1) * rows_in_segment
        training_features_batch = self._training_features[segment_from:segment_to, ]
        training_labels_batch = self._training_labels[segment_from:segment_to]
        return training_features_batch, training_labels_batch

    def forward_propagation(self, training_dataset_batch, training_labels_batch):
        accuracy = self.classification_accuracy(features=training_dataset_batch,
                                                labels=training_labels_batch)
        return accuracy

    def backward_propagation(self, labels):
        self.layers[-1].calculate_dCost_dY(y=labels)
        for layer in self.layers[1:-1][::-1]:
            layer.calculate_dCost_dY()
        for layer in self.layers[1:][::-1]:
            layer.accumulate_weights()
            layer.accumulate_bias()
            layer.apply_accumulations()

    def ensure_layers_have_weights_and_bias(self):
        for layer in self.layers[1:]:
            if layer.bias is None:
                layer.randomise_bias()
            if layer.weights is None:
                layer.randomise_weights()

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

    def load_weights_and_biases(self, filename):
        with open(filename, 'r') as F:
            weights_and_bias_data = json.load(F)
        for data in weights_and_bias_data:
            self.layers[data['layer']].weights = np.array(data['weights'])
            self.layers[data['layer']].bias = np.array(data['bias'])

    def _link_layers(self):
        for indx, layer in enumerate(self.layers[1:]):
            layer.set_previous_layer(self.layers[indx])
