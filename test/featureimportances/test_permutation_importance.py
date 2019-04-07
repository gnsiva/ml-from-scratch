import unittest

import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.featureimportances.permutation_importance import PermutationImportance


class PermutationImportanceTest(unittest.TestCase):
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

        pi = PermutationImportance(rfc, n_iter=10, random_state=42)
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

        pi = PermutationImportance(rfc, n_iter=10, random_state=42)
        pi = pi.fit(test[X_cols], test[y_col])

        df = pi.results_df.copy()

        self.assertGreater(df[df["Feature"] == "petal_width"].iloc[0].Weight, .15)
        self.assertGreater(df[df["Feature"] == "petal_length"].iloc[0].Weight, .15)
        self.assertLess(df[df["Feature"] == "sepal_length"].iloc[0].Weight, .1)
        self.assertLess(df[df["Feature"] == "sepal_width"].iloc[0].Weight, .1)

    def test_show(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        pi = PermutationImportance(rfc, n_iter=10, random_state=42).fit(test[X_cols], test[y_col])

        pi.show()

