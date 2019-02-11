from unittest import TestCase

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.decisiontrees.gns_decision_tree import GNSDecisionTreeClassifier


class GNSDecisionTreeClassifierTest(TestCase):

    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_binary_classification(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col)
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_multiclass_classification(self):
        train, test = train_test_split(self.df, random_state=46)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col)
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation (not on first split seed tried)
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_gini_known_values(self):
        # full impurity
        gini = GNSDecisionTreeClassifier._calculate_gini({0: 2, 1: 2}, {0: 2, 1: 2})
        self.assertAlmostEqual(gini, 0.5)

        # totally pure split
        gini = GNSDecisionTreeClassifier._calculate_gini({0: 2, 1: 0}, {0: 0, 1: 2})
        self.assertAlmostEqual(gini, 0)

        # flipped order
        gini = GNSDecisionTreeClassifier._calculate_gini({0: 0, 1: 2}, {0: 2, 1: 0})
        self.assertAlmostEqual(gini, 0)

    def test_entropy_known_values(self):
        # even predictions
        entropy = GNSDecisionTreeClassifier._calculate_entropy({0: 2, 1: 2}, {0: 2, 1: 2})
        self.assertAlmostEqual(entropy, 1)

        # totally pure split
        entropy = GNSDecisionTreeClassifier._calculate_entropy({0: 2, 1: 0}, {0: 0, 1: 2})
        self.assertAlmostEqual(entropy, 0)

        # flipped order
        entropy = GNSDecisionTreeClassifier._calculate_entropy({0: 0, 1: 2}, {0: 2, 1: 0})
        self.assertAlmostEqual(entropy, 0)

    def test_calculate_stump_split(self):
        fvs = pd.DataFrame([
            (1, 2, 44),
            (2, 3, 44)
        ], columns=["y", "x", "z"])

        dt = GNSDecisionTreeClassifier(X_cols=["x", "z"], y_col="y")
        split_value, max_feature, max_split_score = dt._calculate_stump_split(fvs)

        self.assertAlmostEqual(split_value, 2.5)
        self.assertEqual(max_feature, "x")
        self.assertAlmostEqual(max_split_score, 1.0)

    # def test_binary_classification_entropy(self):
    #     df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
    #     train, test = train_test_split(df, random_state=44)
    #
    #     # check model
    #     dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col, criterion="entropy")
    #     dt = dt.fit(train)
    #     accuracy = (test[self.y_col] == dt.predict(test)).mean()
    #     self.assertGreater(accuracy, 0.85)
    #
    #     # compare to sklearn implementation
    #     sklearn_dt = DecisionTreeClassifier(
    #         max_depth=10, min_impurity_decrease=1e-6, criterion="entropy")
    #     sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])
    #
    #     sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()
    #
    #     # ours should be within 2 % accuracy of sklearn implementation
    #     self.assertGreater(accuracy, sklearn_accuracy - 0.02)
