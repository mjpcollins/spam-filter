import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.layer import Layer


class TestLayer(TestCase):

    def setUp(self):
        self.x = np.array([[0, 1, 0],
                           [0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])
        self.y = np.array([0, 1, 1, 0])
        self.input_layer = Layer(nodes=3)
        self.hidden_layer = Layer(nodes=2,
                                  previous_layer=self.input_layer,
                                  weights=np.array([[0, 0, 0],
                                                    [0, 0, 0]]),
                                  bias=np.array([[0],
                                                 [0]]))
        self.output_layer = Layer(nodes=1,
                                  previous_layer=self.hidden_layer,
                                  weights=np.array([[0, 0]]),
                                  bias=np.array([[0]]))

    def test_init(self):
        self.assertEqual(None, self.input_layer.previous_layer)
        self.assertEqual(self.input_layer, self.hidden_layer.previous_layer)
        self.assertEqual(self.hidden_layer, self.output_layer.previous_layer)

        self.assertEqual(self.hidden_layer, self.input_layer.next_layer)
        self.assertEqual(self.output_layer, self.hidden_layer.next_layer)
        self.assertEqual(None, self.output_layer.next_layer)

        assert_array_equal(None, self.input_layer.weights)
        assert_array_equal(np.array([[0, 0, 0],
                                     [0, 0, 0]]), self.hidden_layer.weights)
        assert_array_equal(np.array([[0, 0]]), self.output_layer.weights)

        assert_array_equal(None, self.input_layer.bias)
        assert_array_equal(np.array([[0],
                                     [0]]), self.hidden_layer.bias)
        assert_array_equal(np.array([[0]]), self.output_layer.bias)

        assert_array_equal(np.array([[0], [0], [0]]), self.input_layer.activation)

    def test_len(self):
        self.assertEqual(3, len(self.input_layer))
        self.assertEqual(2, len(self.hidden_layer))
        self.assertEqual(1, len(self.output_layer))

    def test_activate_all_true(self):
        self.hidden_layer.weights = np.array([[1, 1, 1],
                                              [1, 1, 1]])
        self.hidden_layer.bias = np.zeros([2, 1])
        previous_layer = np.array([[1],
                                   [1],
                                   [1]])
        expected_activation = np.array([[0.9525741268224334],
                                        [0.9525741268224334]])
        actual_activation = self.hidden_layer.activate(previous_layer_activation=previous_layer)
        assert_array_equal(expected_activation, actual_activation)

    def test_and_statement(self):
        input_layer = Layer(nodes=2)
        output_layer = Layer(nodes=1,
                             previous_layer=input_layer,
                             weights=np.array([[10, 10]]),
                             bias=np.array([-15]))

        input_activation = np.array([[1],
                                     [1]])
        expected_activation = np.array([[0.99]])
        actual_activation = output_layer.activate(previous_layer_activation=input_activation)
        assert_array_equal(expected_activation, actual_activation.round(2))

        input_activation = np.array([[0],
                                     [1]])
        expected_activation = np.array([[0.01]])
        actual_activation = output_layer.activate(previous_layer_activation=input_activation)
        assert_array_equal(expected_activation, actual_activation.round(2))

        input_activation = np.array([[1],
                                     [0]])
        expected_activation = np.array([[0.01]])
        actual_activation = output_layer.activate(previous_layer_activation=input_activation)
        assert_array_equal(expected_activation, actual_activation.round(2))

        input_activation = np.array([[0],
                                     [0]])
        expected_activation = np.array([[0.0]])
        actual_activation = output_layer.activate(previous_layer_activation=input_activation)
        assert_array_equal(expected_activation, actual_activation.round(2))

    def test_set_previous_layer(self):
        layer1 = Layer(nodes=3)
        layer2 = Layer(nodes=2)
        layer3 = Layer(nodes=1)
        layer2.set_previous_layer(layer1)
        layer3.set_previous_layer(layer2)

        self.assertEqual(None, layer1.previous_layer)
        self.assertEqual(layer1, layer2.previous_layer)
        self.assertEqual(layer2, layer3.previous_layer)

        self.assertEqual(layer2, layer1.next_layer)
        self.assertEqual(layer3, layer2.next_layer)
        self.assertEqual(None, layer3.next_layer)

    def test_calculate_dC_dA(self):
        pass
