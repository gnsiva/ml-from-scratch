from unittest import TestCase

import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm
import pandas as pd
from sklearn.datasets import load_iris

from ml.naivebayes.gaussian_naive_bayes import GaussianNaiveBayes, Gaussian


class TestGaussianNaiveBayes(TestCase):
    def test_sklearn_example(self):
        """Test adapted from:
        https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.naive_bayes.GaussianNB.html
        """
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])

        clf = GaussianNB()
        clf.fit(X, y)
        print(clf.predict([[-0.8, -1]]))
        print(clf.predict_proba([[-0.8, -1]]))
        print(clf.predict_proba([[-0.8, -1]]).sum())

        nb = GaussianNaiveBayes()
        nb = nb.fit(X, y)

        predictions = nb.predict([[-0.8, -1]])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], 1)

        print(nb.predict([[-0.8, -1]]))
        print(nb.predict_proba([[-0.8, -1]]))  #
        # print(nb.conditional_probabilities)  # shape looks ok
        # print(nb.classes)  # looks good

    def test_iris_example(self):
        """Test adapted from:
        https://medium.com/@awantikdas/a-comprehensive-naive-bayes-tutorial-using-scikit-learn-f6b71ae84431
        """
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        gnb = GaussianNB()
        gnb.fit(df, iris.target)

        gnb.score(df, iris.target)
        print(gnb.predict(df))
        gnb.predict_proba(df)

        nb = GaussianNaiveBayes()
        nb.fit(df, iris.target)
        nb.predict_proba(df)
        print(nb.predict(df))

    def test_conditional_probabilities_simple(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        n_classes = 2
        n_features = 2

        nb = GaussianNaiveBayes()
        nb = nb.fit(X, y)
        self.assertEqual(len(nb.conditional_probabilities), n_classes)
        for v in nb.conditional_probabilities.values():
            self.assertEqual(len(v), n_features)

    def test_conditional_probabilities_iris(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        n_features = 4
        n_classes = 3

        nb = GaussianNaiveBayes()
        nb = nb.fit(df, iris.target)

        self.assertEqual(len(nb.conditional_probabilities), n_classes)
        for v in nb.conditional_probabilities.values():
            self.assertEqual(len(v), n_features)

    def test_gaussian(self):
        x = 4
        mu = 0
        sigma = 1

        scipy_answer = norm(mu, sigma).pdf(x)
        my_answer = Gaussian.gaussian(mu, sigma, x)
        self.assertAlmostEqual(scipy_answer, my_answer)

        mus = np.random.uniform(-4, 4, 10)
        sigmas = np.random.uniform(0, 4, 10)
        xs = np.random.uniform(-2, 2, 10)

        for mu, sigma, x in zip(mus, sigmas, xs):
            scipy_answer = norm(mu, sigma).pdf(x)
            my_answer = Gaussian.gaussian(mu, sigma, x)
            self.assertAlmostEqual(scipy_answer, my_answer)

