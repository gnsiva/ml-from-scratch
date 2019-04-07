from typing import Union, Callable, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

"""
Notes:
======

Permutations, common values:
----------------------------
An example of the issue would be a boolean feature, where 80% of the time it is false.
For a training example which is also false, only 20% of the time you are sampling other values
This could mess with the scoring

Is it ok to do permutation importance on the dev set?
-----------------------------------------------------


Resources:
----------

A git book with good detail with pros and cons of FI
https://christophm.github.io/interpretable-ml-book/feature-importance.html

Straightforward explanation of how it works
https://www.kaggle.com/dansbecker/permutation-importance

The Eli5 page on it
https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html


Untested:
---------
> The first number in each row shows how much model performance decreased with a random shuffling 
> (in this case, using "accuracy" as the performance metric).
https://www.kaggle.com/dansbecker/permutation-importance (bottom)


"""


class PermutationImportance2:
    def __init__(
            self,
            model: BaseEstimator,
            n_iter: int = 5,
            random_state: Optional[int] = None):
        self.model = model
        self.n_iter = n_iter
        self.scoring_function = None
        self.seed = random_state

        # internal parameters
        self.results_df = None

    def _convert_pandas_to_numpy(self, arr):
        if isinstance(arr, pd.DataFrame):
            return arr.values
        return arr

    def _get_columns(self, X, feature_names):
        if isinstance(X, pd.DataFrame):
            return X.columns
        else:
            if feature_names is None:
                return range(X.shape[1])
            else:
                return feature_names

    def _calculate_feature_score(self, X, y, i, original_score):
        data = self._convert_pandas_to_numpy(X).copy()
        scores = np.zeros(self.n_iter, dtype=np.float64)

        for j in range(self.n_iter):
            sampled_feature = np.random.choice(
                data[:, i], size=len(data), replace=False)

            data[:, i] = sampled_feature
            scores[j] = original_score - self._score(data, y)

        return scores.mean(), scores.std()

    def _score(self, X, y):
        return self.model.score(X, y)

    def fit(self, X, y, feature_names=None):
        np.random.seed(self.seed)

        original_score = self._score(X, y)
        feature_names = self._get_columns(X, feature_names)

        results = []
        for i, feature in enumerate(feature_names):
            mean, std = self._calculate_feature_score(X, y, i, original_score)
            results.append((mean, std, feature))
        self.results_df = pd.DataFrame(results, columns=["Weight", "Std", "Feature"])\
            .sort_values(by="Weight", ascending=False)

        return self

    def show(self):
        def print_row(row):
            col = row["Feature"]
            print("{:.3f} +/- {:.3f} - {}".format(row["Weight"], row["Std"], col))

        self.results_df.apply(print_row, axis=1)


class PermutationImportance:
    def __init__(
            self,
            model: BaseEstimator,
            n_iter: int = 5,
            random_state: Optional[int] = None):

        self.model = model
        self.n_iter = n_iter
        self.scoring_function = None
        np.random.seed(random_state)

        # internal parameters
        self.results = []

    # def set_scoring_function(self, metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]):
    #     if isinstance(metric, str):
    #         if metric == "mse":

    def _fit_feature(
            self,
            X: np.ndarray,
            y: np.ndarray,
            i: int,
            bootstrapping: bool = True) -> Tuple[float, float, int]:
        """
        Parameters
        ----------
        X
        y
        i
            Column index for feature to alter

        Returns
        -------
        Feature index, mean score, score standard deviation
        """
        feature = X[:, i].copy()
        if bootstrapping:
            sampled_feature = np.random.choice(
                feature, size=(feature.shape[0], self.n_iter), replace=True)
        else:
            sampled_feature = np.zeros((feature.shape[0], self.n_iter), dtype=feature.dtype)
            for j in range(self.n_iter):
                sampled_feature[:, j] = np.random.choice(
                    feature, size=feature.shape[0], replace=False)

        scores = []
        for j in range(self.n_iter):
            X[:, i] = sampled_feature[:, j]
            if self.scoring_function is not None:
                score = self.scoring_function(self.model.predict(X), y)
            else:
                score = self.model.score(X, y)
            scores.append(score)

        X[:, i] = feature

        return np.mean(scores), np.std(scores), i

    def fit(self, X, y):
        self.results = []
        for i in range(X.shape[1]):
            self.results.append(self._fit_feature(X=X, y=y, i=i))

    def get_results_df(self, feature_names: Optional[List[str]] = None):
        if not len(self.results):
            raise ValueError("Fit first")

        if feature_names is not None:
            if len(self.results) != len(feature_names):
                raise ValueError("")

        df = pd.DataFrame(self.results, columns=["Score", "Stdev", "Feature Index"])
        if feature_names:
            df = df.sort_values(by="Feature Index")
            df["Feature"] = feature_names

        return df.sort_values(by="Score", ascending=False)

    def show(self, feature_names: Optional[List[str]] = None):
        df = self.get_results_df(feature_names=feature_names)

        def print_row(row):
            col = row["Feature Index"]
            if "Feature" in row:
                col = row["Feature"]
            print("{:.3f} +/- {:.3f} - {}".format(row["Score"], row["Stdev"], col))

        df.apply(print_row, axis=1)