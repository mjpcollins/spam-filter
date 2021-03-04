from numpy.testing import assert_array_equal
from unittest import TestCase
from utils.misc import *


class TestMisc(TestCase):

    def test_sigmoid(self):
        self.assertEqual(0.9933071490757153, sigmoid(5))
        self.assertEqual(0.9525741268224334, sigmoid(3))
        self.assertEqual(0.9999999979388463, sigmoid(20))
        self.assertEqual(0.5, sigmoid(0))
        self.assertEqual(0.04742587317756678, sigmoid(-3))

    def test_vectorised_sigmoid(self):
        input_array = np.array([5,
                                0,
                                -3])
        expected_array = np.array([0.9933071490757153,
                                   0.5,
                                   0.04742587317756678])
        actual_array = vectorised_sigmoid(input_array)
        assert_array_equal(expected_array, actual_array)

    def test_prediction_accuracy_function(self):
        self.assertEqual(0.25, cost_function(0.5, 1))
        self.assertEqual(1, cost_function(0, 1))
        self.assertEqual(0, cost_function(1, 1))
        self.assertEqual(0, cost_function(0, 0))
        self.assertEqual(1, cost_function(1, 0))

    def test_vectorised_prediction_accuracy_function(self):
        predictions = np.array([1,
                                0.5,
                                0])
        labels = np.array([0,
                           1,
                           0])
        expected_array = np.array([1,
                                   0.25,
                                   0])
        actual_array = vectorised_cost_function(predictions, labels)
        assert_array_equal(expected_array, actual_array)

    def test_sigmoid_prime(self):
        self.assertEqual(0.045176659730912, sigmoid_prime(3))
        self.assertEqual(2.0611536879193953e-09, sigmoid_prime(20))
        self.assertEqual(0.25, sigmoid_prime(0))
        self.assertEqual(0.04517665973091214, sigmoid_prime(-3))

    def test_vectorised_sigmoid_prime(self):
        input_array = np.array([3,
                                20,
                                0,
                                -3])
        expected_array = np.array([0.045176659730912,
                                   2.0611536879193953e-09,
                                   0.25,
                                   0.04517665973091214])
        actual_array = vectorised_sigmoid_prime(input_array)
        assert_array_equal(expected_array, actual_array)

    def test_cost_function_prime(self):
        self.assertEqual(-1, cost_function_prime(0.5, 1))
        self.assertEqual(-2, cost_function_prime(0, 1))
        self.assertEqual(0, cost_function_prime(1, 1))
        self.assertEqual(0, cost_function_prime(0, 0))
        self.assertEqual(2, cost_function_prime(1, 0))

    def test_vectorised_cost_function_prime(self):
        predictions = np.array([1,
                                0.5,
                                0])
        labels = np.array([0,
                           1,
                           0])
        expected_array = np.array([2,
                                   -1,
                                   0])
        actual_array = vectorised_cost_function_prime(predictions, labels)
        assert_array_equal(expected_array, actual_array)
