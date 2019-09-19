import numpy as np
import pandas as pd
import math

from collections import defaultdict
from typing import Union, Dict, Any, List, DefaultDict


class DiscreteNaiveBayes:
    def __init__(self, sm_param: int = 1):
        """
        Parameters
        ----------
        sm_param:
            Smoothing parameter (0 for no smoothing, but can cause numerical errors).
        """
        self.sm_param = sm_param
        self.column_p_Bs: List[Dict[Any, int]] = []
        self.column_p_B_ys: List[Dict[Any, int]] = []
        self.p_A: float = None

    @staticmethod
    def _calculate_column_p_B(x: np.ndarray) -> DefaultDict[Any, int]:
        values, counts = np.unique(x, return_counts=True)
        p_B = defaultdict(float, {v: c for v, c in zip(values, counts / len(x))})
        return p_B

    @staticmethod
    def _calculate_column_p_B_y(x: np.ndarray, y: Union[np.ndarray, pd.Series]) -> DefaultDict[Any, int]:
        values, counts = np.unique(x[y], return_counts=True)
        p_B_y = defaultdict(float, {v: c for v, c in zip(values, counts / len(x))})
        return p_B_y

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[np.ndarray, pd.Series]):
        self.column_p_Bs = []
        self.column_p_B_ys = []

        self.p_A = y.mean()

        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(X.shape[1]):
            p_B = self._calculate_column_p_B(X[:, i])
            p_B_y = self._calculate_column_p_B_y(X[:, i], y)
            self.column_p_Bs.append(p_B)
            self.column_p_B_ys.append(p_B_y)

        return self

    # TODO - has log transformation, needs the smoothing factor
    # def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
    #     if isinstance(X, pd.DataFrame):
    #         X = X.values
    #
    #     predicted_probabilities = []
    #     for row in X:
    #         log_p_B_y = 0
    #         log_p_B = 0
    #         for v, p_B_d, p_B_y_d in zip(row, self.column_p_Bs, self.column_p_B_ys):
    #             log_p_B_y += math.log(p_B_y_d[v])
    #             log_p_B += math.log(p_B_d[v])
    #
    #         log_p_y_B = log_p_B_y + math.log(self.p_A) - log_p_B
    #         predicted_probabilities.append(math.exp(log_p_y_B))
    #
    #     return predicted_probabilities

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predicted_probabilities = []
        for row in X:
            p_B_y = 1
            p_B = 1
            for v, p_B_d, p_B_y_d in zip(row, self.column_p_Bs, self.column_p_B_ys):
                p_B_y *= p_B_y_d[v]
                p_B *= p_B_d[v]

            p_y_B = (p_B_y * self.p_A) / p_B
            predicted_probabilities.append(p_y_B)

        return predicted_probabilities

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.array([int(round(p)) for p in probs])
