import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.layer import Layer


class TestLayer(TestCase):

    def setUp(self):
        np.random.seed(1)
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

        self.input_layer_2 = Layer(nodes=2)
        self.hidden_layer_2 = Layer(nodes=3,
                                    previous_layer=self.input_layer_2,
                                    weights=np.array([[0.1, 0.6],
                                                      [0.2, 0.4],
                                                      [0.3, 0.7]]),
                                    bias=np.array([[0.],
                                                   [0.],
                                                   [0.]]))
        self.output_layer_2 = Layer(nodes=1,
                                    previous_layer=self.hidden_layer_2,
                                    weights=np.array([[0.1, 0.4, 0.9]]),
                                    bias=np.array([[0.]]))

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

    def test_activate_all_true(self):
        self.hidden_layer.weights = np.array([[1, 1, 1],
                                              [1, 1, 1]])
        self.hidden_layer.bias = np.zeros([2, 1])
        previous_layer = np.array([[1],
                                   [1],
                                   [1]])
        expected_activation = np.array([[0.9525741268224334],
                                        [0.9525741268224334]])
        self.input_layer.activation = previous_layer
        actual_activation = self.hidden_layer.activate()
        assert_array_equal(expected_activation, actual_activation)

    def test_and_statement(self):
        input_layer = Layer(nodes=2)
        output_layer = Layer(nodes=1,
                             previous_layer=input_layer,
                             weights=np.array([[10, 10]]),
                             bias=np.array([-15]))

        expected_activation = np.array([[0.99]])
        input_layer.activation = np.array([[1],
                                           [1]])
        actual_activation = output_layer.activate()
        assert_array_equal(expected_activation, actual_activation.round(2))

        expected_activation = np.array([[0.01]])
        input_layer.activation = np.array([[0],
                                           [1]])
        actual_activation = output_layer.activate()
        assert_array_equal(expected_activation, actual_activation.round(2))

        expected_activation = np.array([[0.01]])
        input_layer.activation = np.array([[1],
                                           [0]])
        actual_activation = output_layer.activate()
        assert_array_equal(expected_activation, actual_activation.round(2))

        expected_activation = np.array([[0.0]])
        input_layer.activation = np.array([[0],
                                           [0]])
        actual_activation = output_layer.activate()
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

    def test_activate(self):
        self._activate_layers_2()
        expected_activation = np.array([[0.668, 0.712, 0.688, 0.728]])
        assert_array_equal(expected_activation, self.output_layer_2.activation.round(3))

    def test_cost(self):
        self._activate_layers_2()
        labels = np.array([[0., 1., 1., 0.]])
        self.assertEqual(0.145, round(self.output_layer_2.cost(y=labels), 3))

    def test_dCost_dY_output_layer(self):
        self._activate_layers_2()
        expected_dCost_dY = np.array([[0.167, -0.072, -0.078, 0.182]])
        assert_array_equal(expected_dCost_dY, self.output_layer_2.dCost_dY.round(3))

    def test_dY_dZ_output_layer(self):
        self._activate_layers_2()
        expected_dY_dZ = np.array([[0.222, 0.205, 0.215, 0.198]])
        assert_array_equal(expected_dY_dZ, self.output_layer_2.dY_dZ().round(3))

    def test_dZ_dW_output_layer(self):
        self._activate_layers_2()
        expected_dZ_dW = np.array([[0.5, 0.5, 0.5],
                                   [0.646, 0.599, 0.668],
                                   [0.525, 0.55, 0.574],
                                   [0.668, 0.646, 0.731]])
        assert_array_equal(expected_dZ_dW, self.output_layer_2.dZ_dW().round(3))

    def test_dZ_dB_output_layer(self):
        self._activate_layers_2()
        expected_dZ_dB = np.array([[1.]])
        assert_array_equal(expected_dZ_dB, self.output_layer_2.dZ_dB().round(3))

    def test_dZ_dA_output_layer(self):
        self._activate_layers_2()
        expected_dZ_dA = np.array([[0.1],
                                   [0.4],
                                   [0.9]])
        assert_array_equal(expected_dZ_dA, self.output_layer_2.dZ_dA().round(3))

    def test_dCost_dZ_output_layer(self):
        self._activate_layers_2()
        expected_dCost_dZ = np.array([[0.037, -0.015, -0.017, 0.036]])
        assert_array_equal(expected_dCost_dZ, self.output_layer_2.dCost_dZ().round(3))

    def test_dCost_dW_output_layer(self):
        self._activate_layers_2()
        expected_dCost_dW = np.array([[0.024, 0.024, 0.025]])
        assert_array_equal(expected_dCost_dW, self.output_layer_2.dCost_dW().round(3))

    def test_dCost_dB_output_layer(self):
        self._activate_layers_2()
        expected_dCost_dB = np.array(0.042)
        assert_array_equal(expected_dCost_dB, self.output_layer_2.dCost_dB().round(3))

    def test_calculate_dCost_dY_hidden_layer(self):
        self._activate_layers_2()
        expected_dCost_dY = np.array([[0.004, -0.001, -0.002, 0.004],
                                      [0.015, -0.006, -0.007, 0.014],
                                      [0.033, -0.013, -0.015, 0.032]])
        assert_array_equal(expected_dCost_dY, self.hidden_layer_2.dCost_dY.round(3))

    def test_dY_dZ_hidden_layer(self):
        self._activate_layers_2()
        expected_dY_dZ = np.array([[0.25, 0.229, 0.249, 0.222],
                                   [0.25, 0.24, 0.248, 0.229],
                                   [0.25, 0.222, 0.244, 0.197]])
        assert_array_equal(expected_dY_dZ, self.hidden_layer_2.dY_dZ().round(3))

    def test_dCost_dZ_hidden_layer(self):
        self._activate_layers_2()
        expected_dCost_dZ = np.array([[0.001, 0., 0., 0.001],
                                      [0.004, -0.001, -0.002, 0.003],
                                      [0.008, -0.003, -0.004, 0.006]])
        assert_array_equal(expected_dCost_dZ, self.hidden_layer_2.dCost_dZ().round(3))

    def test_dZ_dW_hidden_layer(self):
        self._activate_layers_2()
        expected_dZ_dW = np.array([[0., 0.],
                                   [0., 1.],
                                   [1., 0.],
                                   [1., 1.]])
        assert_array_equal(expected_dZ_dW, self.hidden_layer_2.dZ_dW().round(3))

    def test_dZ_dB_hidden_layer(self):
        self._activate_layers_2()
        expected_dZ_dB = np.array([[1.],
                                   [1.],
                                   [1.]])
        assert_array_equal(expected_dZ_dB, self.hidden_layer_2.dZ_dB().round(3))

    def test_dCost_dW_hidden_layer(self):
        self._activate_layers_2()
        expected_dCost_dW = np.array([[0.0004, 0.0005],
                                      [0.0016, 0.0019],
                                      [0.0027, 0.0034]])
        assert_array_equal(expected_dCost_dW, self.hidden_layer_2.dCost_dW().round(4))

    def test_dCost_dB_hidden_layer(self):
        self._activate_layers_2()
        expected_dCost_dB = np.array([[0.001],
                                      [0.004],
                                      [0.008]])
        assert_array_equal(expected_dCost_dB, self.hidden_layer_2.dCost_dB().round(3))

    def test_update_all(self):
        self._activate_layers_2()
        self.output_layer_2.accumulate_weights()
        self.output_layer_2.accumulate_bias()
        self.hidden_layer_2.accumulate_weights()
        self.hidden_layer_2.accumulate_bias()
        self.output_layer_2.apply_accumulations()
        self.hidden_layer_2.apply_accumulations()

        expected_output_w = np.array([[0.076, 0.376, 0.875]])
        expected_output_b = np.array([[-0.042]])
        expected_hidden_w = np.array([[0.1, 0.6],
                                      [0.198, 0.398],
                                      [0.297, 0.697]])
        expected_hidden_b = np.array([[-0.001],
                                      [-0.004],
                                      [-0.008]])

        assert_array_equal(expected_output_w, self.output_layer_2.weights.round(3))
        assert_array_equal(expected_output_b, self.output_layer_2.bias.round(3))
        assert_array_equal(expected_hidden_w, self.hidden_layer_2.weights.round(3))
        assert_array_equal(expected_hidden_b, self.hidden_layer_2.bias.round(3))

    def test_update_all_second_iteration(self):
        for _ in range(2):
            self._activate_layers_2()
            self.output_layer_2.accumulate_weights()
            self.output_layer_2.accumulate_bias()
            self.hidden_layer_2.accumulate_weights()
            self.hidden_layer_2.accumulate_bias()
            self.output_layer_2.apply_accumulations()
            self.hidden_layer_2.apply_accumulations()

        expected_output_dCost_dW = np.array([[0.023, 0.022, 0.024]])
        expected_output_dCost_dB = np.array([[0.039]])
        expected_hidden_dCost_dW = np.array([[0, 0],
                                             [0.001, 0.002],
                                             [0.002, 0.003]])
        expected_hidden_dCost_dB = np.array([[0.001],
                                             [0.003],
                                             [0.007]])

        assert_array_equal(expected_output_dCost_dW, self.output_layer_2.dCost_dW().round(3))
        assert_array_equal(expected_output_dCost_dB, self.output_layer_2.dCost_dB().round(3))
        assert_array_equal(expected_hidden_dCost_dW, self.hidden_layer_2.dCost_dW().round(3))
        assert_array_equal(expected_hidden_dCost_dB, self.hidden_layer_2.dCost_dB().round(3))

    def test_5000_epochs(self):
        for _ in range(5000):
            self._activate_layers_2()
            self.output_layer_2.accumulate_weights()
            self.output_layer_2.accumulate_bias()
            self.hidden_layer_2.accumulate_weights()
            self.hidden_layer_2.accumulate_bias()
            self.output_layer_2.apply_accumulations()
            self.hidden_layer_2.apply_accumulations()
        self.input_layer_2.activation = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.hidden_layer_2.activate()
        self.output_layer_2.activate()

        expected_activation = np.array([[0.05, 0.96, 0.96, 0.04]])
        expected_output_layer_2_weights = np.array([[-6.53535281, -4.72686643, 8.778615]])
        expected_output_layer_2_bias = np.array([[-3.3564161]])
        expected_hidden_layer_2_weights = np.array([[3.12306195, 3.16183072],
                                                    [2.21900195, 2.18280679],
                                                    [6.11392827, 6.13281431]])
        expected_hidden_layer_2_bias = np.array([[-4.9099341],
                                                 [-3.50174506],
                                                 [-2.70728321]])

        self.assertEqual(0.0009, round(self.output_layer_2.cost(y=np.array([[0., 1., 1., 0.]])), 4))
        assert_array_equal(expected_activation, self.output_layer_2.activation.round(2))
        assert_array_equal(expected_output_layer_2_weights, self.output_layer_2.weights.round(8))
        assert_array_equal(expected_output_layer_2_bias, self.output_layer_2.bias.round(8))
        assert_array_equal(expected_hidden_layer_2_weights, self.hidden_layer_2.weights.round(8))
        assert_array_equal(expected_hidden_layer_2_bias, self.hidden_layer_2.bias.round(8))

    def test_randomise_weights(self):
        self.hidden_layer_2.randomise_weights()
        self.output_layer_2.randomise_weights()
        expected_hidden_layer_2_weights = np.array([[0.47532, 0.74829],
                                                    [0.1001,  0.3721],
                                                    [0.23208, 0.1831]])
        expected_output_layer_2_weights = np.array([[0.26763, 0.411, 0.45709]])
        assert_array_equal(expected_hidden_layer_2_weights, self.hidden_layer_2.weights.round(5))
        assert_array_equal(expected_output_layer_2_weights, self.output_layer_2.weights.round(5))

    def test_randomise_bias(self):
        self.hidden_layer_2.randomise_bias()
        self.output_layer_2.randomise_bias()
        expected_hidden_layer_2_bias = np.array([[0.47532],
                                                 [0.74829],
                                                 [0.1001]])
        expected_output_layer_2_bias = np.array([[0.3721]])
        assert_array_equal(expected_hidden_layer_2_bias, self.hidden_layer_2.bias.round(5))
        assert_array_equal(expected_output_layer_2_bias, self.output_layer_2.bias.round(5))

    def test_randomise_and_activate(self):
        self.hidden_layer_2.randomise_weights()
        self.output_layer_2.randomise_weights()
        self.hidden_layer_2.randomise_bias()
        self.output_layer_2.randomise_bias()
        self.input_layer_2.activation = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.hidden_layer_2.activate()
        self.output_layer_2.activate()
        assert_array_equal(np.array([[0.73, 0.75, 0.75, 0.76]]), self.output_layer_2.activation.round(2))

    def _activate_layers_2(self):
        self.input_layer_2.activation = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.hidden_layer_2.activate()
        self.output_layer_2.activate()
        self.output_layer_2.calculate_dCost_dY(y=np.array([[0., 1., 1., 0.]]))
        self.hidden_layer_2.calculate_dCost_dY()
