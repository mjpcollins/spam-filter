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
                          [0, 1, 0, 1]])
        expected_prediction = np.array([[0.05, 0.96, 0.96, 0.04]])
        assert_array_equal(expected_prediction, nn.predict(input).round(2))

    def test_train_xor(self):
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]])
        labels = np.array([[0., 1., 1., 0.]])
        self.nn.set_training_data(dataset, labels)
        self.nn.train(epochs=5000)
        expected_prediction = np.array([[0.06, 0.94, 0.94, 0.06]])
        assert_array_equal(expected_prediction, self.nn.predict(dataset).round(2))

    def test_set_training_data(self):
        dataset = np.array([[0., 0., 1., 1.],
                            [0., 1., 0., 1.]])
        labels = np.array([[0., 1., 1., 0.]])
        self.nn.set_training_data(dataset, labels)
        assert_array_equal(dataset, self.nn.training_data)
        assert_array_equal(labels, self.nn.training_labels)
