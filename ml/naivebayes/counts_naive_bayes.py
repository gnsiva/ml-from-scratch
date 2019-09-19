from collections import defaultdict
from typing import Tuple, Union

import numpy as np
import pandas as pd


class CountsNaiveBayes:
    def __init__(self):
        self.n_features: int
        self.priors = {}
        self.conditional_probabilities = defaultdict(list)
        self.classes = None

    def _calc_priors(self, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        total_rows = len(y)
        return {cls: cnt / total_rows for cls, cnt in zip(classes, counts)}

    def _pandas_to_np(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        else:
            return X
