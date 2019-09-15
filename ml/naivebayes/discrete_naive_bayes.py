import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any, List


class DiscreteNaiveBayes:
    def __init__(self, normalising_factor: int = 1):
        self.normalising_factor = normalising_factor
        self.column_p_Bs: List[Dict[Any, int]] = []
        self.column_p_B_ys: List[Dict[Any, int]] = []
        self.p_A: float = None

    @staticmethod
    def _fit_column(x: np.ndarray, y: Union[np.ndarray, pd.Series]) -> Tuple[Dict[Any, int], Dict[Any, int]]:
        values, counts = np.unique(x, return_counts=True)
        p_B =  {v: c for v, c in zip(values, counts / len(x))}

        values, counts = np.unique(x[y], return_counts=True)
        p_B_y = {v: c for v, c in zip(values, counts / len(x))}

        return p_B, p_B_y

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[np.ndarray, pd.Series]):
        self.column_p_Bs = []
        self.column_p_B_ys = []

        self.p_A = y.mean()

        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(X.shape[1]):
            p_B, p_B_y = self._fit_column(X[:, i], y)
            self.column_p_Bs.append(p_B)
            self.column_p_B_ys.append(p_B_y)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        predicted_probabilities = []
        for row in X:
            for v, p_B_d, p_B_y_d in zip(row, self.column_p_Bs, self.column_p_B_ys):
                p = (p_B_y_d[v] * self.p_A) / p_B_d[v]
                predicted_probabilities.append(p)

        return predicted_probabilities


