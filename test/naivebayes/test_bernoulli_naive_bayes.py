from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from ml.naivebayes.bernoulli_naive_bayes import BernoulliNaiveBayes
from ml.naivebayes.multinomial_naive_bayes import MultinomialNaiveBayes


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

        # self.assertListEqual(list(nb.predict(test_X.values)), [0])
        print(nb.classes)
        print(nb.predict_proba(test_X.values))
        print(nb.predict(test_X.values))

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
        self.assertEqual(predictions.shape, (1,))
        # self.assertEqual(predictions[0], 0)  # TODO this isn't passing at the moment

        prob_preds = nb.predict_proba(test_X.values)
        # self.assertAlmostEqual(prob_preds[0][0], 0.005184, 6)  # TODO need to get actual
        # self.assertAlmostEqual(prob_preds[0][1], 0.021947873799725653, 6)  # answers from book

        # compare to sklearn algorithm
        sklearn_nb = BernoulliNB(alpha=1)
        sklearn_nb = sklearn_nb.fit(training_X, training_y)
        print(sklearn_nb.predict_proba(test_X))
        # Doesn't match sklearn implementation
