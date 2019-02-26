import logging
import math
from typing import Tuple, Dict
from scipy import stats

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
#from joblib import Parallel, delayed

"""
Notes:
======
- Maybe you should only optimise for min_samples_split or min_samples_leaf, but not both
- Look into what oob_score actually does

"""

class RandomForestRegressor:
    def __init__(
            self,
            n_estimators: int = 10,
            criterion: str = "mse",
            max_features: float = 1.0,
            bagging: bool = True):

        # RF parameters
        self.n_estimators = n_estimators
        self.bagging = bagging
        self.max_features = max_features

        # DT parameters
        self.tree_parameters = {
            'criterion': criterion
        }

        # Internal parameters
        self.trees = []
        self.tree_type = "regressor"
        self.ensemble_reduce_function = "mean"

    @staticmethod
    def _subsample_features(X, max_features: int) -> np.ndarray:
        n_columns = X.shape[1]
        indices = np.arange(n_columns)
        if n_columns == max_features:
            return X, indices
        else:
            indices = np.random.choice(indices, max_features, replace=False)
            return X[:, indices], indices

    @staticmethod
    def _sampling(X, y, bagging: bool) -> Tuple[np.ndarray, np.ndarray]:
        if bagging:
            n = len(X)
            indices = np.random.choice(np.arange(n), n, replace=True)
            if isinstance(X, pd.DataFrame):
                return X.iloc[indices], y[indices]
            else:
                return X[indices], y[indices]
        return X, y

    @staticmethod
    def _calculate_max_features_int(n_columns: int, max_features: float) -> int:
        n_features = round(n_columns * max_features)
        if n_features > n_columns:
            msg = "max_features too high, results in {} columns when only {} are present".format(
                    n_features, n_columns)

            msg += "\nUsing {} features instead".format(n_columns)
            logging.warning(msg)
            return n_columns
        return n_features

    @staticmethod
    def _fit_a_tree(
            X, y,
            n_features: int,
            bagging: bool,
            tree_type: str,
            tree_parameters: Dict) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Parameters
        ----------
        X
        y
        n_features
            number of features to subsample
        bagging
            whether to sample with replacement or not
        tree_type
            should be 'regressor' or classifier'
        tree_parameters
            passed to decision tree

        Returns
        -------
        1. The fitted tree
        2. The indices of the subsampled features used
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X, feature_indices = RandomForestRegressor._subsample_features(X, n_features)
        X, y = RandomForestRegressor._sampling(X, y, bagging)

        if tree_type == "regressor":
            dt = DecisionTreeRegressor(**tree_parameters)
        elif tree_type == "classifier":
            dt = DecisionTreeClassifier(**tree_parameters)
        else:
            raise ValueError("Unsupported tree type: {}".format(tree_type))

        return dt.fit(X, y), feature_indices

    def fit(self, X, y):
        n_features = self._calculate_max_features_int(X.shape[1], self.max_features)

        self.trees = []
        for i in range(self.n_estimators):
            tree_feature_indices = self._fit_a_tree(
                X=X,
                y=y,
                n_features=n_features,
                bagging=self.bagging,
                tree_type=self.tree_type,
                tree_parameters=self.tree_parameters
            )
            self.trees.append(tree_feature_indices)

        return self

    @staticmethod
    def _index_pandas_columns(df: pd.DataFrame, indices) -> pd.DataFrame:
        return df.iloc[:, indices]

    @staticmethod
    def _index_numpy_columns(arr: np.array, indices) -> np.array:
        return arr[:, indices]

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))

        if isinstance(X, pd.DataFrame):
            index_function = RandomForestRegressor._index_pandas_columns
        else:
            index_function = RandomForestRegressor._index_numpy_columns

        for tree, feature_indices in self.trees:
            p = tree.predict(index_function(X, feature_indices))
            predictions[:, 0] = p

        if self.ensemble_reduce_function == "mode":
            mode_result = stats.mode(predictions, axis=1)
            output = mode_result[0].flatten()
        elif self.ensemble_reduce_function == "mean":
            output = np.mean(predictions, axis=1)

        return output
