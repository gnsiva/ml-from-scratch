import seaborn as sns
from unittest import TestCase

from ml.decisiontrees.random_forest import RandomForestRegressor as RFR, RandomForestClassifier as RFC


class RandomForestRegressorTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_simple_fit_runs(self):
        rfr = RFR(
            n_estimators=10,
            criterion="mse",
            max_features=0.5,
            bagging=True)

        rfr = rfr.fit(self.df[self.X_cols], self.df[self.y_col])

        p = rfr.predict(self.df[self.X_cols])
        self.assertEqual(p.shape, self.df[self.y_col].shape)

    def test__calculate_max_features_int(self):
        self.assertEqual(RFR._calculate_max_features_int(n_columns=10, max_features=0.5), 5)
        self.assertEqual(RFR._calculate_max_features_int(n_columns=10, max_features=1.5), 10)


class RandomForestClassifierTest(TestCase):
    def setUp(self):
        df = sns.load_dataset("iris")
        df.loc[df["species"] == "virginica", "species_i"] = 0
        df.loc[df["species"] == "versicolor", "species_i"] = 1
        df.loc[df["species"] == "setosa", "species_i"] = 2

        self.df = df
        self.X_cols = df.columns[:-2].tolist()
        self.y_col = "species_i"

    def test_simple_fit_runs(self):
        rfr = RFC(
            n_estimators=10,
            criterion="gini",
            max_features=0.5,
            bagging=True)

        rfr = rfr.fit(self.df[self.X_cols], self.df[self.y_col])

        p = rfr.predict(self.df[self.X_cols])
        self.assertEqual(p.shape, self.df[self.y_col].shape)


