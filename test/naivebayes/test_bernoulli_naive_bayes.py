from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

from ml.naivebayes.bernoulli_naive_bayes import BernoulliNaiveBayes


class TestBernoulliNaiveBayes(TestCase):
    def test_book_example(self):
        training_X = pd.DataFrame({
            "Chinese": [True, True, True, True],
            "Beijing": [True, False, False, False],
            "Shanghai": [False, True, False, False],
            "Macao": [False, False, True, False],
            "Tokyo": [False, False, False, True],
            "Japan": [False, False, False, True],
        })
        # is related to china
        training_y = np.array([1, 1, 1, 0])

        test_X = pd.DataFrame({
            "Chinese": [True],
            "Beijing": [False],
            "Shanghai": [False],
            "Macao": [False],
            "Tokyo": [True],
            "Japan": [True],
        })

        nb = BernoulliNaiveBayes()
        nb = nb.fit(training_X.values, training_y)

        # check prediction
        self.assertEqual(nb.predict(test_X.values)[0], 0)

        # check conditional probs
        for obs, exp in zip(nb.conditional_probabilities[0],
            [2/3, 1/3, 1/3, 1/3, 2/3, 2/3]):
            self.assertAlmostEqual(obs, exp)

        for obs, exp in zip(nb.conditional_probabilities[1],
            [0.8, 0.4, 0.4, 0.4, 0.2, 0.2]):
            self.assertAlmostEqual(obs, exp)

        # check priors
        self.assertAlmostEqual(nb.priors[0], 0.25)
        self.assertAlmostEqual(nb.priors[1], 0.75)

        predictions = nb.predict(test_X.values)
        self.assertEqual(len(predictions), 1)

        prob_preds = nb.predict_proba(test_X.values)
        self.assertAlmostEqual(prob_preds[0][1], 0.005184, 6)
        self.assertAlmostEqual(prob_preds[0][0], 0.021947873799725653, 6)

    def test_book_example_sklearn(self):
        training_X = pd.DataFrame({
            "Chinese": [True, True, True, True],
            "Beijing": [True, False, False, False],
            "Shanghai": [False, True, False, False],
            "Macao": [False, False, True, False],
            "Tokyo": [False, False, False, True],
            "Japan": [False, False, False, True],
        })
        # is related to china
        training_y = np.array([1, 1, 1, 0])

        test_X = pd.DataFrame({
            "Chinese": [True],
            "Beijing": [False],
            "Shanghai": [False],
            "Macao": [False],
            "Tokyo": [True],
            "Japan": [True],
        })

        nb = BernoulliNaiveBayes(use_p_B=True)
        nb = nb.fit(training_X.values, training_y)

        # check prediction
        self.assertEqual(nb.predict(test_X.values)[0], 0)

        # check conditional probs
        for obs, exp in zip(nb.conditional_probabilities[0],
            [2/3, 1/3, 1/3, 1/3, 2/3, 2/3]):
            self.assertAlmostEqual(obs, exp)

        for obs, exp in zip(nb.conditional_probabilities[1],
            [0.8, 0.4, 0.4, 0.4, 0.2, 0.2]):
            self.assertAlmostEqual(obs, exp)

        # check priors
        self.assertAlmostEqual(nb.priors[0], 0.25)
        self.assertAlmostEqual(nb.priors[1], 0.75)

        predictions = nb.predict(test_X.values)
        self.assertEqual(len(predictions), 1)

        prob_preds = nb.predict_proba(test_X.values)

        # compare to sklearn algorithm
        sklearn_nb = BernoulliNB(alpha=1)
        sklearn_nb = sklearn_nb.fit(training_X, training_y)
        sk_proba = sklearn_nb.predict_proba(test_X)

        for my_row, sk_row in zip(prob_preds, sk_proba):
            for my, sk in zip(my_row.values(), sk_row):
                self.assertAlmostEqual(my, sk, 5)


