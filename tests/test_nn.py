import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.nn import NeuralNetwork


class TestLayer(TestCase):

    def setUp(self):
        np.random.seed(1)
        self.nn = NeuralNetwork(layer_setup=[3, 2, 1])

    def test_init(self):
        self.assertEqual(None, self.nn.layers[0].previous_layer)
        self.assertEqual(self.nn.layers[0], self.nn.layers[1].previous_layer)
        self.assertEqual(self.nn.layers[1], self.nn.layers[2].previous_layer)

    def test_predict_all_weights_zero(self):
        input_array = np.array([[1], [1], [0]])
        expected_result = np.array([[0.75]])
        actual_result = self.nn.predict(input_array)
        assert_array_equal(expected_result, actual_result.round(2))
