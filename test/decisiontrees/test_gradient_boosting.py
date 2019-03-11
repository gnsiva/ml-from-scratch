import seaborn as sns
from unittest import TestCase

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from ml.decisiontrees.gradient_boosting import GradientBoostingRegressor as GBR


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
