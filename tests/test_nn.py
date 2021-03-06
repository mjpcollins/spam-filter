import os
import json
import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.nn import NeuralNetwork


class TestLayer(TestCase):

    def setUp(self):
        np.random.seed(1)
        layer_settings = [{'nodes': 2},
                          {'nodes': 3},
                          {'nodes': 1}]
        self.nn = NeuralNetwork(layer_settings=layer_settings)

    def test_init(self):
        self.assertEqual(None, self.nn.layers[0].previous_layer)
        self.assertEqual(self.nn.layers[0], self.nn.layers[1].previous_layer)
        self.assertEqual(self.nn.layers[1], self.nn.layers[2].previous_layer)

    def test_predict_xor(self):
        output_layer_weights = np.array([[-6.53535281, -4.72686643, 8.778615]])
        output_layer_bias = np.array([[-3.3564161]])
        hidden_layer_weights = np.array([[3.12306195, 3.16183072],
                                         [2.21900195, 2.18280679],
                                         [6.11392827, 6.13281431]])
        hidden_layer_bias = np.array([[-4.9099341],
                                      [-3.50174506],
                                      [-2.70728321]])
        layer_settings = [{'nodes': 2},
                          {'nodes': 3, 'weights': hidden_layer_weights, 'bias': hidden_layer_bias},
                          {'nodes': 1, 'weights': output_layer_weights, 'bias': output_layer_bias}]
        nn = NeuralNetwork(layer_settings=layer_settings)
        input = np.array([[0, 0, 1, 1],
                          [0, 1, 0, 1]]).T
        expected_prediction = np.array([[0., 1., 1., 0.]])
        assert_array_equal(expected_prediction, nn.predict(input).round(2))

    def test_train_xor(self):
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]]).T
        labels = np.array([[0., 1., 1., 0.]])
        self.nn.set_training_data(dataset, labels)
        self.nn.train(epochs=5000)
        expected_prediction = np.array([[0., 1., 1., 0.]])
        assert_array_equal(expected_prediction, self.nn.predict(dataset).round(2))

    def test_set_training_data(self):
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]])
        labels = np.array([[0., 1., 1., 0.]])
        self.nn.set_training_data(dataset, labels)
        assert_array_equal(dataset, self.nn._training_data)
        assert_array_equal(labels, self.nn._training_labels)

    def test_classification_accuracy(self):
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]]).T
        labels = np.array([[0., 1., 1., 0.]])
        self.nn.set_training_data(dataset, labels)
        self.nn.train(epochs=5000)
        worse_labels = np.array([[0., 1., 1., 1.]])
        self.assertEqual(1, self.nn.classification_accuracy(dataset=dataset,
                                                            labels=labels).round(2))
        self.assertEqual(0.75, self.nn.classification_accuracy(dataset=dataset,
                                                               labels=worse_labels).round(2))

    def test_save_weights_and_biases(self):
        output_layer_weights = np.array([[-6.53535281, -4.72686643, 8.778615]])
        output_layer_bias = np.array([[-3.3564161]])
        hidden_layer_weights = np.array([[3.12306195, 3.16183072],
                                         [2.21900195, 2.18280679],
                                         [6.11392827, 6.13281431]])
        hidden_layer_bias = np.array([[-4.9099341],
                                      [-3.50174506],
                                      [-2.70728321]])
        layer_settings = [{'nodes': 2},
                          {'nodes': 3, 'weights': hidden_layer_weights, 'bias': hidden_layer_bias},
                          {'nodes': 1, 'weights': output_layer_weights, 'bias': output_layer_bias}]
        nn = NeuralNetwork(layer_settings=layer_settings)
        save_file = "tests/save_test.json"
        self.try_rm_save_data(filename=save_file)
        nn.save_weights_and_biases(filename=save_file)
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
        expected_output_layer_weights = np.array([[-6.53535281, -4.72686643, 8.778615]])
        expected_output_layer_bias = np.array([[-3.3564161]])
        expected_hidden_layer_weights = np.array([[3.12306195, 3.16183072],
                                                  [2.21900195, 2.18280679],
                                                  [6.11392827, 6.13281431]])
        expected_hidden_layer_bias = np.array([[-4.9099341],
                                               [-3.50174506],
                                               [-2.70728321]])
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]]).T
        expected_prediction = np.array([[0., 1., 1., 0.]])

        assert_array_equal(expected_output_layer_weights, self.nn.layers[2].weights)
        assert_array_equal(expected_output_layer_bias, self.nn.layers[2].bias)
        assert_array_equal(expected_hidden_layer_weights, self.nn.layers[1].weights)
        assert_array_equal(expected_hidden_layer_bias, self.nn.layers[1].bias)
        assert_array_equal(expected_prediction, self.nn.predict(dataset).round(2))

    @staticmethod
    def try_rm_save_data(filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
