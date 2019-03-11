import logging
from typing import Dict, Optional

from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradientBoostingRegressor:
    def __init__(
            self,
            learning_rate: float = 0.1,
            criterion: str = "mse",
            subsample_fraction: float = 1.0,
            max_features: float = 1.0,
            n_estimators: int = 10,
            min_score_improvement: float = 1e-4,
            tree_params: Optional[Dict] = None):

        self.learning_rate = learning_rate
        self.criterion = criterion
        self.subsample_fraction = subsample_fraction
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_score_improvement = min_score_improvement

        self.tree_params = {}
        if tree_params:
            self.tree_params = tree_params

        # internal parameters
        self.trees = []
        self.y0 = None

    def fit(self, X, y):
        self.trees = []

        self.y0 = y.mean()
        yhat = np.full(X.shape[0], self.y0)
        residuals = y - yhat
        mse = (residuals ** 2).sum()

        for i in range(self.n_estimators):
            # fit a tree on the residuals
            dt = DecisionTreeRegressor(**self.tree_params)
            dt = dt.fit(X, residuals)

            # update the total prediction
            yhat += self.learning_rate * dt.predict(X)
            residuals = y - yhat

            # check stopping criteria
            new_mse = (residuals ** 2).sum()
            if (mse - new_mse) < self.min_score_improvement:
                logging.debug("Reached minimum score improvement, exiting on estimator {}".format(i + 1))
                break

            mse = new_mse
            self.trees.append(dt)

        return self

    def predict(self, X):
        yhat = np.full(X.shape[0], self.y0)
        for tree in self.trees:
            yhat += self.learning_rate * tree.predict(X)

        return yhat
