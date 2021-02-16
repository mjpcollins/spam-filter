import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from classifiers import DecisionTree


class TestDecisionTree(TestCase):

    def setUp(self):
        self.small_training_data = np.array([[1, 1, 1, 1, 0],
                                             [1, 1, 1, 0, 0],
                                             [1, 1, 1, 0, 0],
                                             [1, 1, 0, 0, 0],
                                             [0, 0, 1, 0, 1],
                                             [0, 0, 0, 1, 1],
                                             [0, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 1]])
        self.labels = self.small_training_data[:, 0]
        self.features = self.small_training_data[:, 1:]
        self.decision_node = DecisionTree(features=self.features,
                                          labels=self.labels)

    def test_calculate_lowest_gini_feature(self):
        actual_gini = self.decision_node._lowest_gini_feature_noise()
        self.assertEqual(0, actual_gini)

    def test_calculate_lowest_gini_feature_with_exclusions(self):
        self.decision_node._exclusions = {0, 3}
        actual_gini = self.decision_node._lowest_gini_feature_noise()
        self.assertEqual(1, actual_gini)

    def test_calculate_lowest_gini_feature_with_all_exclusions(self):
        self.decision_node._exclusions = {0, 1, 2, 3}
        actual_gini = self.decision_node._lowest_gini_feature_noise()
        self.assertEqual(None, actual_gini)

    def test_predict(self):
        test_data = np.array([[1, 1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 1]])
        self.assertEqual(1, self.decision_node.predict(test_data[0, :]))
        self.assertEqual(1, self.decision_node.predict(test_data[1, :]))
        self.assertEqual(0, self.decision_node.predict(test_data[2, :]))

    def test_predict_with_complex_labels(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        test_data = np.array([[1, 1, 0, 0],
                              [1, 1, 0, 0],
                              [0, 1, 0, 1],
                              [1, 0, 0, 1]])
        decision_node = DecisionTree(features=self.features,
                                     labels=labels)
        self.assertEqual(1, decision_node.predict(test_data[0, :]))
        self.assertEqual(1, decision_node.predict(test_data[1, :]))
        self.assertEqual(0, decision_node.predict(test_data[2, :]))
        self.assertEqual(0, decision_node.predict(test_data[3, :]))

    def test_calculate_gini_index(self):
        self.assertEqual(0, self.decision_node._calculate_gini_index(self.features[:, 0]))
        self.assertEqual(0.375, self.decision_node._calculate_gini_index(self.features[:, 1]))
        self.assertEqual(0.4375, self.decision_node._calculate_gini_index(self.features[:, 2]))
        self.assertEqual(0, self.decision_node._calculate_gini_index(self.features[:, 3]))