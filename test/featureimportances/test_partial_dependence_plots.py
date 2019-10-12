import unittest

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from ml.featureimportances.partial_dependence_plots import PartialDependencePlots


class PartialDependencePlotsTests(unittest.TestCase):

    def _get_basic_data(self, regression=False):
        df = sns.load_dataset("iris")
        if regression:
            df = df[df["species"].isin(["setosa", "versicolor"])].copy()

        df["species_i"] = df.species.astype("category").cat.codes

        X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        y_col = "species_i"

        train, test = train_test_split(df, random_state=4)
        return train, test, X_cols, y_col

    # ========================================= classification tests

    def setup_data(self):
        train, test, X_cols, y_col = self._get_basic_data()

        rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
        rfc = rfc.fit(train[X_cols], train[y_col])

        return train.reset_index(drop=True), test.reset_index(drop=True), X_cols, y_col, rfc

    def test_calculate_partial_predictions(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        pdp = PartialDependencePlots(rfc, X_cols)

        mean_predictions, stdev_predictions = pdp.calculate_partial_predictions(test[X_cols], "petal_width")

        print(mean_predictions)
        # TODO - at least do a test to show all preds aren't the same

    def test_do_partial_plot(self):
        train, test, X_cols, y_col, rfc = self.setup_data()
        feature = "petal_width"

        pdp = PartialDependencePlots(rfc, X_cols)
        pdp.do_partial_plot(test[X_cols], feature)
        # import matplotlib.pyplot as plt
        # plt.show()

    # ========================================= regression tests

    def setup_regression_data(self):
        train, test, X_cols, y_col = self._get_basic_data(regression=True)

        rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
        rfr = rfr.fit(train[X_cols], train[y_col])

        return train.reset_index(drop=True), test.reset_index(drop=True), X_cols, y_col, rfr

    def test_calculate_partial_predictions_regression(self):
        train, test, X_cols, y_col, rfr = self.setup_regression_data()

        pdp = PartialDependencePlots(rfr, X_cols)

        mean_predictions, stdev_predictions = pdp.calculate_partial_predictions(test[X_cols], "petal_width")

        print(mean_predictions)

    def test_do_partial_plot_regression(self):
        train, test, X_cols, y_col, rfr = self.setup_regression_data()
        feature = "petal_width"

        pdp = PartialDependencePlots(rfr, X_cols)
        pdp.do_partial_plot(test[X_cols], feature)
        import matplotlib.pyplot as plt
        # plt.show()
