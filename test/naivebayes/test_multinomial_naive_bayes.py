from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from ml.naivebayes.multinomial_naive_bayes import MultinomialNaiveBayes


class TestMultinomialNaiveBayes(TestCase):
    def test_book_example(self):
        training_X = pd.DataFrame({
            "Chinese": [2, 2, 1, 1],
            "Beijing": [1, 0, 0, 0],
            "Shanghai": [0, 1, 0, 0],
            "Macao": [0, 0, 1, 0],
            "Tokyo": [0, 0, 0, 1],
            "Japan": [0, 0, 0, 1],
        })
        # is related to china
        training_y = np.array([1, 1, 1, 0])

        test_X = pd.DataFrame({
            "Chinese": [3],
            "Beijing": [0],
            "Shanghai": [0],
            "Macao": [0],
            "Tokyo": [1],
            "Japan": [1],
        })

        nb = MultinomialNaiveBayes()
        nb = nb.fit(training_X.values, training_y)
        prob_preds = nb.predict_proba(test_X.values)
        self.assertAlmostEqual(prob_preds[0][0], 0.0001354807, 6)
        self.assertAlmostEqual(prob_preds[0][1], 0.0003012137, 6)

        self.assertListEqual(list(nb.predict(test_X.values)), [1])
        print(nb.predict_proba(test_X.values))

        # compare to sklearn algorithm
        sklearn_nb = MultinomialNB(alpha=1)
        sklearn_nb = sklearn_nb.fit(training_X, training_y)
        print(sklearn_nb.predict_proba(test_X))
        # Doesn't match sklearn implementation
