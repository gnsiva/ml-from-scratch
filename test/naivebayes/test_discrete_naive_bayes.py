from unittest import TestCase

import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from ml.naivebayes.discrete_naive_bayes import DiscreteNaiveBayes


class TestDiscreteNaiveBayes(TestCase):
    def test__calculate_column_p_B(self):
        p_B = DiscreteNaiveBayes._calculate_column_p_B([1, 1, 1])
        self.assertEqual(len(p_B), 1)
        self.assertAlmostEqual(p_B[1], 1, 5)

        p_B = DiscreteNaiveBayes._calculate_column_p_B([1, 1, 1, 2])
        self.assertEqual(len(p_B), 2)
        self.assertAlmostEqual(p_B[1], .75, 5)
        self.assertAlmostEqual(p_B[2], .25, 5)

        p_B = DiscreteNaiveBayes._calculate_column_p_B([1, 1, 2, 3])
        self.assertEqual(len(p_B), 3)
        self.assertAlmostEqual(p_B[1], .5, 5)
        self.assertAlmostEqual(p_B[2], .25, 5)
        self.assertAlmostEqual(p_B[3], .25, 5)

    def test__calculate_column_p_B_y(self):
        pass

    def test_compare_to_sklearn_very_simple(self):
        data = pd.DataFrame([
            (True, 1, 1),
            (False, 2, 2),
            (True, 3, 1)
        ], columns=["y", "col1", "col2"])

        dnb = DiscreteNaiveBayes()
        dnb = dnb.fit(data[["col1", "col2"]], data["y"])
        print(dnb.predict(data[["col1", "col2"]]))

        # compare to MultinomialNB
        X = pd.concat([pd.get_dummies(data["col1"]), pd.get_dummies(data["col2"])], axis=1)

        nb = MultinomialNB(alpha=0)
        nb = nb.fit(X, data["y"])
        print(nb.predict(X))

        self.assertListEqual(
            dnb.predict(data[["col1", "col2"]]).tolist(),
            nb.predict(X).astype(int).tolist()
        )

        # compare to BernoulliNB
        bnb = BernoulliNB(alpha=0)
        bnb = bnb.fit(X, data["y"])
        print(bnb.predict(X))

        self.assertListEqual(
            dnb.predict(data[["col1", "col2"]]).tolist(),
            bnb.predict(X).astype(int).tolist()
        )

        print(dnb.predict_proba(data[["col1", "col2"]]))
        print(nb.predict_proba(X))
        print(bnb.predict_proba(X))
