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
