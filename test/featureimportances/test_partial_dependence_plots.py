import unittest

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.featureimportances.partial_dependence_plots import PartialDependencePlots


class PartialDependencePlotsTests(unittest.TestCase):
    def setup_data(self):
        df = sns.load_dataset("iris")
        df["species_i"] = df.species.astype("category").cat.codes

        X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        y_col = "species_i"

        train, test = train_test_split(df, random_state=4)

        rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
        rfc = rfc.fit(train[X_cols], train[y_col])

        return train.reset_index(drop=True), test.reset_index(drop=True), X_cols, y_col, rfc

    def test_calculate_partial_predictions(self):
        train, test, X_cols, y_col, rfc = self.setup_data()

        pdp = PartialDependencePlots(rfc, X_cols)

        mean_predictions, stdev_predictions = pdp.calculate_partial_predictions(test[X_cols], "petal_width")

        print(mean_predictions)


