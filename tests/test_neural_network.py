import os
import json
import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.neural_network import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def setUp(self):
        layer_settings = [{'nodes': 2},
                          {'nodes': 3},
                          {'nodes': 1}]

        self.xor_features = np.array([[0, 0, 1, 1],
                                      [0, 1, 0, 1]]).T
        self.xor_labels = np.array([[0, 1, 1, 0]])
        self.xor_output_layer_weights = np.array([[-6.53535281, -4.72686643, 8.778615]])
        self.xor_output_layer_bias = np.array([[-3.3564161]])
        self.xor_hidden_layer_weights = np.array([[3.12306195, 3.16183072],
                                                  [2.21900195, 2.18280679],
                                                  [6.11392827, 6.13281431]])
        self.xor_hidden_layer_bias = np.array([[-4.9099341],
                                               [-3.50174506],
                                               [-2.70728321]])
        xor_layer_settings = [{'nodes': 2},
                              {'nodes': 3, 'weights': self.xor_hidden_layer_weights, 'bias': self.xor_hidden_layer_bias},
                              {'nodes': 1, 'weights': self.xor_output_layer_weights, 'bias': self.xor_output_layer_bias}]

        self.nn = NeuralNetwork(layer_settings=layer_settings)
        self.xor_nn = NeuralNetwork(layer_settings=xor_layer_settings)

    def test_init(self):
        self.assertEqual(None, self.nn.layers[0].previous_layer)
        self.assertEqual(self.nn.layers[0], self.nn.layers[1].previous_layer)
        self.assertEqual(self.nn.layers[1], self.nn.layers[2].previous_layer)

    def test_predict_xor(self):
        assert_array_equal(self.xor_labels, self.xor_nn.predict(self.xor_features).round(2))

    def test_train_xor(self):
        self.nn.set_training_data(self.xor_features,
                                  self.xor_labels)
        self.nn.train(epochs=3000)
        assert_array_equal(self.xor_labels, self.nn.predict(self.xor_features).round(2))

    def test_set_training_data(self):
        self.nn.set_training_data(self.xor_features,
                                  self.xor_labels)
        assert_array_equal(self.xor_features, self.nn._training_features)
        assert_array_equal(self.xor_labels, self.nn._training_labels)

    def test_classification_accuracy(self):
        self.nn.set_training_data(self.xor_features,
                                  self.xor_labels)
        self.nn.train(epochs=3000)
        self.assertEqual(1, self.nn.classification_accuracy(features=self.xor_features,
                                                            labels=self.xor_labels).round(2))
        bad_labels = np.array([[0., 1., 1., 1.]])
        self.assertEqual(0.75, self.nn.classification_accuracy(features=self.xor_features,
                                                               labels=bad_labels).round(2))

    def test_save_weights_and_biases(self):
        save_file = "tests/save_test.json"
        self.try_rm_save_data(filename=save_file)
        self.xor_nn.save_weights_and_biases(filename=save_file)
        with open(save_file, 'r') as F:
            saved_data = json.load(F)
        expected_saved_data = [{"layer":  0,
                                "weights": [[]],
                                "bias": [[]]},
                               {"layer":  1,
                                "weights": [[3.12306195, 3.16183072],
                                            [2.21900195, 2.18280679],
                                            [6.11392827, 6.13281431]],
                                "bias":[[-4.9099341],
                                        [-3.50174506],
                                        [-2.70728321]]},
                               {"layer":  2,
                                "weights": [[-6.53535281, -4.72686643, 8.778615]],
                                "bias":[[-3.3564161]]}]
        self.assertEqual(expected_saved_data, saved_data)
        self.try_rm_save_data(filename=save_file)

    def test_load_weights_and_biases(self):
        self.nn.load_weights_and_biases(filename="tests/xor_weights_and_biases.json")
        assert_array_equal(self.xor_output_layer_weights, self.nn.layers[2].weights)
        assert_array_equal(self.xor_output_layer_bias, self.nn.layers[2].bias)
        assert_array_equal(self.xor_hidden_layer_weights, self.nn.layers[1].weights)
        assert_array_equal(self.xor_hidden_layer_bias, self.nn.layers[1].bias)
        assert_array_equal(self.xor_labels, self.nn.predict(self.xor_features).round(2))

    def test_get_training_batch(self):
        training_full_dataset = np.genfromtxt("data/training_spam.csv", delimiter=',')
        training_features = training_full_dataset[:, 1:]
        training_labels = training_full_dataset[:, 0]
        layer_settings = [{'nodes': 54},
                          {'nodes': 3},
                          {'nodes': 1}]
        nn = NeuralNetwork(layer_settings)
        nn.set_training_data(features=training_features,
                             labels=training_labels)
        actual_features_batch_0, actual_labels_batch_0 = nn.get_training_batch(batch=0,
                                                                               rows_in_segment=200)
        actual_features_batch_1, actual_labels_batch_1 = nn.get_training_batch(batch=1,
                                                                               rows_in_segment=200)
        expected_features_batch_0 = training_features[0:200, :]
        expected_features_batch_1 = training_features[200:400, :]
        expected_labels_batch_0 = training_labels[0:200]
        expected_labels_batch_1 = training_labels[200:400]

        assert_array_equal(actual_features_batch_0, expected_features_batch_0)
        assert_array_equal(actual_features_batch_1, expected_features_batch_1)
        assert_array_equal(actual_labels_batch_0, expected_labels_batch_0)
        assert_array_equal(actual_labels_batch_1, expected_labels_batch_1)

    @staticmethod
    def try_rm_save_data(filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
