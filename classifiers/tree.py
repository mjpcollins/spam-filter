import numpy as np


class DecisionTree:

    def __init__(self, features, labels, exclusions=None, depth=0, max_depth=4):
        if not exclusions:
            self._exclusions = []
        else:
            self._exclusions = exclusions
        self._depth = depth + 1
        self._max_depth = max_depth
        self._labels = labels
        self._features = features
        self.feature_decision = None
        self.true = None
        self.false = None
        if self._depth < self._max_depth:
            self._split()

    def __bool__(self):
        return self.feature_decision is not None

    def predict(self, data):
        if data[self.feature_decision]:
            if self.true:
                return self.true.predict(data)
            return 1
        if self.false:
            return self.false.predict(data)
        return 0

    def _split(self):
        self.feature_decision = self._lowest_gini_feature_noise()
        if self:
            filter_true = np.where(self._features[:, self.feature_decision])
            filter_false = np.where(self._features[:, self.feature_decision] == 0)
            if filter_true[0].any():
                self.true = DecisionTree(features=self._features[filter_true],
                                         labels=self._labels[filter_true],
                                         exclusions=self._exclusions + [self.feature_decision],
                                         depth=self._depth,
                                         max_depth=self._max_depth)
            if filter_false[0].any():
                self.false = DecisionTree(features=self._features[filter_false],
                                          labels=self._labels[filter_false],
                                          exclusions=self._exclusions + [self.feature_decision],
                                          depth=self._depth,
                                          max_depth=self._max_depth)

    def _lowest_gini_feature_noise(self):
        noise = []
        for index in range(self._features.shape[1]):
            if index not in self._exclusions:
                noise.append(self._calculate_gini_index(self._features[:, index]))
            else:
                noise.append(2)
        if min(noise) != 2:
            return noise.index(min(noise))
        return None

    def _calculate_gini_index(self, features):
        features1 = features[np.where(self._labels)]
        features0 = features[np.where(self._labels == 0)]
        high_gini_index = 1 - (np.mean(features1)**2 + (1 - np.mean(features1))**2)
        low_gini_index = 1 - (np.mean(features0)**2 + (1 - np.mean(features0))**2)
        return np.mean(self._labels) * high_gini_index + (1 - np.mean(self._labels)) * low_gini_index
