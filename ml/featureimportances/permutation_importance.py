from typing import Union, Optional, Tuple, List

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


class PermutationImportance:
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

    def _convert_pandas_to_numpy(
            self,
            arr: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(arr, pd.DataFrame):
            return arr.values
        return arr

    def _get_columns(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            feature_names: List[str]):

        if isinstance(X, pd.DataFrame):
            return X.columns
        else:
            if feature_names is None:
                return range(X.shape[1])
            else:
                return feature_names

    def _calculate_feature_score(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            i: int,
            original_score: float) -> Tuple[float, float]:
        data = self._convert_pandas_to_numpy(X).copy()
        scores = np.zeros(self.n_iter, dtype=np.float64)

        for j in range(self.n_iter):
            sampled_feature = np.random.choice(
                data[:, i], size=len(data), replace=False)

            data[:, i] = sampled_feature
            scores[j] = original_score - self._score(data, y)

        return scores.mean(), scores.std()

    def _score(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray) -> float:
        return self.model.score(X, y)

    def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            feature_names=None) -> 'PermutationImportance':

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
