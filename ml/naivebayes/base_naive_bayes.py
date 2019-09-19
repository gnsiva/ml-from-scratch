from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, Any, List, Dict

import numpy as np
import pandas as pd


class BaseNaiveBayes(ABC):
    def __init__(self):
        self.priors = {}
        self.conditional_probabilities = defaultdict(list)
        self.predictor_prior: float = None
        self.classes = None

    def _calc_priors(self, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        total_rows = len(y)
        return {cls: cnt / total_rows for cls, cnt in zip(classes, counts)}

    @abstractmethod
    def _calc_conditional_probabilities(self, X, y):
        pass

    @abstractmethod
    def _calc_p_B(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict_proba(self, X) -> List[Dict[Any, float]]:
        X = self._pandas_to_np(X)
        return [self._predict_proba_row(row) for row in X]

    def predict(self, X) -> List[Any]:
        return [max(d, key=d.get) for d in self.predict_proba(X)]

    def _pandas_to_np(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        else:
            return X


class CountsNaiveBayes(BaseNaiveBayes):
    def __init__(self):
        super().__init__()

    # def _calc_priors(self, y):
    #     classes, counts = np.unique(y, return_counts=True)
    #     self.classes = classes
    #     total_rows = len(y)
    #     return {cls: cnt / total_rows for cls, cnt in zip(classes, counts)}


