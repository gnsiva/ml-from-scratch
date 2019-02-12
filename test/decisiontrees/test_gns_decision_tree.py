from unittest import TestCase

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml.decisiontrees.gns_decision_tree import GNSDecisionTreeClassifier, GNSDecisionTreeRegressor


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
        entropy = GNSDecisionTreeClassifier._calculate_entropy(
            pd.Series([0, 0, 1, 1]), pd.Series([0, 0, 1, 1]))
        self.assertAlmostEqual(entropy, 1)

        # totally pure split
        entropy = GNSDecisionTreeClassifier._calculate_entropy(
            pd.Series([0, 0]), pd.Series([1, 1]))
        self.assertAlmostEqual(entropy, 0)

        # flipped order
        entropy = GNSDecisionTreeClassifier._calculate_entropy(
            pd.Series([1, 1]), pd.Series([0, 0]))
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

    def test_binary_classification_entropy(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col, criterion="entropy")
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(
            max_depth=10, min_impurity_decrease=1e-6, criterion="entropy")
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_multiclass_classification_entropy(self):
        train, test = train_test_split(self.df, random_state=46)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col, criterion="entropy")
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)
        print(accuracy)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=1e-6, criterion="entropy")
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation (not on first split seed tried)
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_binary_classification_gini2(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col, criterion="gini2")
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_multiclass_classification_gini2(self):
        train, test = train_test_split(self.df, random_state=46)

        # check model
        dt = GNSDecisionTreeClassifier(self.X_cols, self.y_col, criterion="gini2")
        dt = dt.fit(train)
        accuracy = (test[self.y_col] == dt.predict(test)).mean()
        self.assertGreater(accuracy, 0.85)

        # compare to sklearn implementation
        sklearn_dt = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])

        sklearn_accuracy = (test[self.y_col] == sklearn_dt.predict(test[self.X_cols])).mean()

        # ours should be within 2 % accuracy of sklearn implementation (not on first split seed tried)
        self.assertGreater(accuracy, sklearn_accuracy - 0.02)

    def test_gini2_known_values(self):
        # full impurity
        gini = GNSDecisionTreeClassifier._calculate_gini2(
            pd.Series([0, 0, 1, 1]), pd.Series([0, 0, 1, 1]))
        self.assertAlmostEqual(gini, 0.5)

        # totally pure split
        gini = GNSDecisionTreeClassifier._calculate_gini2(
            pd.Series([0, 0]), pd.Series([1, 1]))
        self.assertAlmostEqual(gini, 0)

        # flipped order
        gini = GNSDecisionTreeClassifier._calculate_gini2(
            pd.Series([1, 1]), pd.Series([0, 0]))
        self.assertAlmostEqual(gini, 0)


class GNSDecisionTreeRegressorTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_mse(self):
        a = pd.Series([3, 3, 3])
        self.assertEqual(GNSDecisionTreeRegressor._mse(a, a), 0)

        a = pd.Series([2, 4])
        self.assertEqual(GNSDecisionTreeRegressor._mse(a, a), 2)

    def test_prediction_one_var(self):
        df = pd.DataFrame({
            "y": [0, 1, 1],
            "x": [4, 6, 6]
        })
        X_cols = ["x"]

        dt = GNSDecisionTreeRegressor(X_cols, "y")
        dt = dt.fit(df)
        p = dt.predict(df)

        self.assertAlmostEqual(p[0], 0)
        self.assertAlmostEqual(p[1], 1)
        self.assertAlmostEqual(p[2], 1)

        sklearn_dt = DecisionTreeRegressor(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(df[X_cols], df["y"])
        sklearn_p = sklearn_dt.predict(df[X_cols])

        self.assertAlmostEqual(sklearn_p[0], 0)
        self.assertAlmostEqual(sklearn_p[1], 1)
        self.assertAlmostEqual(sklearn_p[2], 1)

    def test_prediction_two_vars(self):
        df = pd.DataFrame({
            "y": [0, 1, 1],
            "x": [4, 6, 6],
            "z": [3, 3, 3]
        })

        X_cols = ["x", "z"]

        dt = GNSDecisionTreeRegressor(X_cols, "y")
        dt = dt.fit(df)
        p = dt.predict(df)

        self.assertAlmostEqual(p[0], 0)
        self.assertAlmostEqual(p[1], 1)
        self.assertAlmostEqual(p[2], 1)

        sklearn_dt = DecisionTreeRegressor(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(df[X_cols], df["y"])
        sklearn_p = sklearn_dt.predict(df[X_cols])

        self.assertAlmostEqual(sklearn_p[0], 0)
        self.assertAlmostEqual(sklearn_p[1], 1)
        self.assertAlmostEqual(sklearn_p[2], 1)

    def test_prediction_real_data(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GNSDecisionTreeRegressor(self.X_cols, self.y_col, max_depth=10)

        dt = dt.fit(train)
        brier_score = ((test[self.y_col] - dt.predict(test))**2).mean()
        # print(brier_score)

        # check against sklearn implementation
        sklearn_dt = DecisionTreeRegressor(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])
        sklearn_p = sklearn_dt.predict(test[self.X_cols])
        sklearn_brier_score = ((test[self.y_col] - sklearn_p) ** 2).mean()

        # print(sklearn_brier_score)

        self.assertLess(brier_score, sklearn_brier_score + 0.02)

    # def test_predict_with_ww_data(self):
    #     fn = "../../books/190122-train-fvs-no-dev-avgs-no-lax3.p.gz"
    #     fvs = pd.read_pickle(fn)
    #     rids = fvs[["rid"]].sample(frac=0.01, random_state=1).rid.tolist()
    #     fvs = fvs[fvs.rid.isin(rids)]
    #     train = fvs[fvs.dataset == "train"]
    #     test = fvs[fvs.dataset != "train"]
    #
    #     X_cols = [
    #         "local_hour",
    #         "weekday",
    #         "num",
    #         "pstid",
    #         'pstid_2_total_num',
    #         'pstid_2_rid_count',
    #         'pstid_4_total_num',
    #         'pstid_4_rid_count',
    #         'total_num',
    #         'total_rid_count'
    #     ]
    #
    #     y_col = "aspace"
    #
    #     dt = GNSDecisionTreeRegressor(X_cols, y_col, max_depth=10)
    #     dt = dt.fit(fvs)
    #     brier_score = ((test[y_col] - dt.predict(test)) ** 2).mean()
    #     print(brier_score)
    #
    #     # check against sklearn implementation
    #     sklearn_dt = DecisionTreeRegressor(max_depth=10, min_impurity_decrease=1e-6)
    #     sklearn_dt = sklearn_dt.fit(train[X_cols], train[y_col])
    #     sklearn_p = sklearn_dt.predict(test[X_cols])
    #     sklearn_brier_score = ((test[y_col] - sklearn_p) ** 2).mean()
    #     print(sklearn_brier_score)
