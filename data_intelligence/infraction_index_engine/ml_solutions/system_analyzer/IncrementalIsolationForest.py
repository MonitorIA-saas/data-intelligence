import numpy as np
from sklearn.ensemble import IsolationForest

class IncrementalIsolationForest:
    def __init__(self, n_estimators=200, max_samples=256, replace_rate=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.replace_rate = replace_rate
        self.random_state = random_state
        self.model = None
        self.ages = []

    def fit(self, X):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.ages = [0] * len(self.model.estimators_)
        return self

    def partial_fit(self, X_new):
        if self.model is None:
            return self.fit(X_new)

        self.ages = [age + 1 for age in self.ages]

        K = max(1, int(self.n_estimators * self.replace_rate))

        new_forest = IsolationForest(
            n_estimators=K,
            max_samples=min(self.max_samples, len(X_new)),
            random_state=self.random_state,
        )
        new_forest.fit(X_new)

        old_estimators = self.model.estimators_
        new_estimators = new_forest.estimators_

        oldest_indices = np.argsort(self.ages)[-K:]

        for i, idx in enumerate(oldest_indices):
            old_estimators[idx] = new_estimators[i]
            self.ages[idx] = 0 

        self.model.estimators_ = old_estimators
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        return self.model.predict(X)