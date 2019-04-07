import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance as Eli5PI
import seaborn as sns
from ppdspy.metrics import brier_score_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ml.featureimportances.permutation_importance import PermutationImportance, PermutationImportance2


def prep_df(df):
    # TODO - add a random column and a numeric y value
    df["random"] = np.random.rand(len(df))

    df.loc[df["species"] == "virginica", "species_i"] = 0
    df.loc[df["species"] == "versicolor", "species_i"] = 1
    df.loc[df["species"] == "setosa", "species_i"] = 2

    y = df["species_i"].copy()
    df = df.drop(columns=["species", "species_i"])
    X = df.values
    return X, y, list(df.columns)


class PermutationImportanceTest(unittest.TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        self.X, self.y, self.X_cols = prep_df(df)
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(self.X, self.y)

    # def test__fit_feature(self):
    #     mock_model = Mock()
    #     mock_model.predict.return_value = np.array([1, 1, 2])
    #     mock_model.score.return_value = np.array([1, 1, 2])
    #     test_X = pd.DataFrame({
    #         'a': [1, 1, 2],
    #         'b': [3, 4, 5]
    #     })
    #
    #     pi = PermutationImportance(
    #         model=mock_model,
    #         n_iter=10,
    #         random_state=5
    #     )
    #
    #     mean_score, std_score, i = pi._fit_feature(test_X.values, np.array([0, 1, 2]), 0, False)
    #     print(mean_score)

    def test__fit_feature(self):
        class MockModel:
            def predict(self, X):
                return (X[:, 0] == 2).astype(np.uint8)

            def score(self, X, y):
                return accuracy_score(self.predict(X), y)

        test_X = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [3, 4, 5]
        })

        pi = PermutationImportance(
            model=MockModel(),
            n_iter=10,
            random_state=5
        )

        mean_score, std_score, i = pi._fit_feature(test_X.values, np.array([0, 0, 1]), 0, False)
        print(mean_score)

        mean_score, std_score, i = pi._fit_feature(test_X.values, np.array([0, 0, 1]), 1, False)
        print(mean_score)

    def test_show(self):
        rfr = RandomForestRegressor(n_estimators=100)
        rfr = rfr.fit(self.train_X, self.train_y)

        pi = PermutationImportance(
            model=rfr,
            n_iter=10,
            random_state=5
        )

        pi.fit(self.test_X, self.test_y)
        pi.show(self.X_cols)


class PermutationImportance2Test(unittest.TestCase):
    def setup_data(self):
        df = sns.load_dataset("iris")
        df["species_i"] = df.species.astype("category").cat.codes

        X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        y_col = "species_i"

        train, test = train_test_split(df, random_state=4)

        rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
        rfc = rfc.fit(train[X_cols], train[y_col])

        return train, test, X_cols, y_col, rfc

    def test__calculate_feature_score(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        original_score = rfc.score(test[X_cols], test[y_col])

        pi = PermutationImportance2(rfc, n_iter=10, random_state=42)
        np.random.seed(42)

        scores_d = {}
        for i in range(4):
            score, std = pi._calculate_feature_score(test[X_cols], test[y_col], i, original_score)
            scores_d[X_cols[i]] = (score, std)

        self.assertGreater(scores_d["petal_width"][0], .15)
        self.assertGreater(scores_d["petal_length"][0], .15)
        self.assertLess(scores_d["sepal_length"][0], .1)
        self.assertLess(scores_d["sepal_width"][0], .1)

    def test_fit(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        pi = PermutationImportance2(rfc, n_iter=10, random_state=42)
        pi = pi.fit(test[X_cols], test[y_col])

        df = pi.results_df.copy()

        self.assertGreater(df[df["Feature"] == "petal_width"].iloc[0].Weight, .15)
        self.assertGreater(df[df["Feature"] == "petal_length"].iloc[0].Weight, .15)
        self.assertLess(df[df["Feature"] == "sepal_length"].iloc[0].Weight, .1)
        self.assertLess(df[df["Feature"] == "sepal_width"].iloc[0].Weight, .1)

    def test_show(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        pi = PermutationImportance2(rfc, n_iter=10, random_state=42).fit(test[X_cols], test[y_col])

        pi.show()

