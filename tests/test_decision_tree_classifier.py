import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from classifiers import DecisionTreeSpamClassifier


class TestDecisionTreeSpamClassifier(TestCase):

    def setUp(self):
        with open("../data/training_spam.csv") as F:
            self.training_data = np.array([[int(value) for value in l.replace("\n", "").split(",")] for l in F.readlines()])
        with open("../data/testing_spam.csv") as F:
            self.testing_data = np.array([[int(value) for value in l.replace("\n", "").split(",")] for l in F.readlines()])
        self.test_labels = self.testing_data[:, 0]
        self.test_features = self.testing_data[:, 1:]
        self.classifier = DecisionTreeSpamClassifier(self.training_data)

    def test_prediction_performance(self):
        for i in range(2, 20):
            classifier = DecisionTreeSpamClassifier(self.training_data, max_depth=i)
            predictions = classifier.predict(self.test_features)
            accuracy = np.count_nonzero(predictions == self.test_labels) / self.test_labels.shape[0]
            print(f"With max_depth set to {i}, accuracy on test data is: {accuracy}")
