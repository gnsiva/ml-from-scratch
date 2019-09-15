from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class PartialDependencePlots:
    def __init__(
            self,
            model: BaseEstimator,
            X_cols: List[str]):
        self.model = model
        self.X_cols = X_cols

        # internal parameters
        self.results_df = None

    def calculate_partial_predictions(self, df: pd.DataFrame, feature: str):
        df = df.copy()
        feature_values = df[feature].values.copy()
        n = len(df)

        predictions = np.zeros((n, n), dtype=float)
        for i in range(n):
            df = df.loc[np.roll(df.index, -1)]
            df[feature] = feature_values

            # each prediction round is a column
            predictions[:, i] = self.model.predict(df[self.X_cols])

        predictions_mean = predictions.mean(axis=1)
        predictions_stdev = predictions.std(axis=1)

        return predictions_mean, predictions_stdev

    def do_partial_plot(self, df: pd.DataFrame, feature: str):
        predictions, stdevs = self.calculate_partial_predictions(df, feature)

        results_df = pd.DataFrame({
            feature: df[feature],
            "prediction": predictions,
            "stdev": stdevs
        }).sort_values(by=feature)

        results_df.plot(x=feature, y="prediction", yerr="stdev", fmt="--o")
        # TODO - test
