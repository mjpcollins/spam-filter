import numpy as np
from classifiers.tree import DecisionTree


class DecisionTreeSpamClassifier:
    def __init__(self, training_data, max_depth=5):
        self._labels = training_data[:, 0]
        self._features = training_data[:, 1:]
        self._max_depth = max_depth
        self._tree = None

    def train(self):
        self._tree = DecisionTree(features=self._features,
                                  labels=self._labels,
                                  max_depth=self._max_depth)

    def predict(self, data):
        if self._tree is None:
            self.train()
        predicitons = []
        for row in data:
            predicitons.append(self._tree.predict(row))
        return np.array(predicitons)

