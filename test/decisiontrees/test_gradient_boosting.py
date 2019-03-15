import seaborn as sns
from unittest import TestCase

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

from ml.decisiontrees.gradient_boosting import GradientBoostingRegressor as GBR, _GBRDecisionTreeRegressor, \
    GradientBoostingMAERegressor


class GradientBoostingRegressorTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_simple_fit_runs(self):
        gbr = GBR(
            n_estimators=10,
            criterion="mse")

        gbr = gbr.fit(self.df[self.X_cols], self.df[self.y_col])

        p = gbr.predict(self.df[self.X_cols])
        self.assertEqual(p.shape, self.df[self.y_col].shape)

    def test_prediction_real_data(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GBR(learning_rate=0.1, tree_params={"max_depth": 4})

        dt = dt.fit(train[self.X_cols], train[self.y_col])
        brier_score = ((test[self.y_col] - dt.predict(test[self.X_cols]))**2).mean()
        print(brier_score)

        # check against sklearn implementation
        sklearn_dt = GradientBoostingRegressor(max_depth=4, n_estimators=10)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])
        sklearn_p = sklearn_dt.predict(test[self.X_cols])
        sklearn_brier_score = ((test[self.y_col] - sklearn_p) ** 2).mean()
        print(sklearn_brier_score)

        self.assertLess(brier_score, sklearn_brier_score + 0.02)


class GradientBoostingMAERegressorTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_simple_fit_runs(self):
        gbr = GradientBoostingMAERegressor(
            X_cols=self.X_cols,
            y_col=self.y_col,
            n_estimators=10,
            criterion="mse")

        # gbr = gbr.fit(self.df[self.X_cols], self.df[self.y_col])
        gbr = gbr.fit(self.df)

        p = gbr.predict(self.df)
        self.assertEqual(p.shape, self.df[self.y_col].shape)

    def test_prediction_real_data(self):
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = GradientBoostingMAERegressor(
            X_cols=self.X_cols,
            y_col=self.y_col,
            learning_rate=0.1,
            tree_params={"max_depth": 4})

        dt = dt.fit(train)
        real_y = test[self.y_col].copy()
        test[self.y_col] = -1

        mae = mean_absolute_error(real_y, dt.predict(test[self.X_cols]))
        test[self.y_col] = real_y
        print(mae)

        # check against sklearn implementation
        sklearn_dt = GradientBoostingRegressor(loss="lad", max_depth=4, n_estimators=10)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])
        sklearn_p = sklearn_dt.predict(test[self.X_cols])
        sklearn_mae = mean_absolute_error(test[self.y_col], sklearn_p)
        print(sklearn_mae)

        self.assertLess(mae, sklearn_mae + 0.02)


class _GBRDecisionTreeRegressorTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_prediction_real_data(self):
        """Make sure the tree still works normally"""
        df = self.df[self.df.species.isin(["versicolor", "virginica"])].copy()
        train, test = train_test_split(df, random_state=44)

        # check model
        dt = _GBRDecisionTreeRegressor(self.X_cols, self.y_col, max_depth=10)

        dt = dt.fit(train)
        brier_score = ((test[self.y_col] - dt.predict(test))**2).mean()
        # print(brier_score)

        # check against sklearn implementation
        sklearn_dt = DecisionTreeRegressor(max_depth=10, min_impurity_decrease=1e-6)
        sklearn_dt = sklearn_dt.fit(train[self.X_cols], train[self.y_col])
        sklearn_p = sklearn_dt.predict(test[self.X_cols])
        sklearn_brier_score = ((test[self.y_col] - sklearn_p) ** 2).mean()

        self.assertLess(brier_score, sklearn_brier_score + 0.02)

    def test_median_final_leaf(self):
        df = pd.DataFrame({
            "residuals": [0, 1, 1, 1],
            "target": [2, 6, 6, 10],
            "x": [4, 6, 6, 6]
        })
        X_cols = ["x"]

        dt = _GBRDecisionTreeRegressor(X_cols, "residuals", "target")
        dt = dt.fit(df)
        p = dt.predict_median_leaf(df, df["target"].median())
        p += df["target"].median()

        self.assertAlmostEqual(p[0], 2)
        self.assertAlmostEqual(p[1], 6)
        self.assertAlmostEqual(p[2], 6)
        self.assertAlmostEqual(p[2], 6)


