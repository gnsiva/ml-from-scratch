from collections import defaultdict

import math

import numpy as np

from ml.naivebayes.base_naive_bayes import BaseNaiveBayes


class Gaussian:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def calc(self, x: float):
        return self.gaussian(self.mu, self.sigma, x)

    @staticmethod
    def gaussian(self, mu, sigma, x):
        coefficent = 1 / (math.sqrt(2 * math.pi * sigma ** 2))
        exponent = - ((x - mu) ** 2) / (2 * sigma ** 2)
        return coefficent * math.exp(exponent)


class GaussianNaiveBayes(BaseNaiveBayes):
    def gaussian(self, mu, sigma, x):
        coefficent = 1 / (math.sqrt(2 * math.pi * sigma ** 2))
        exponent = - ((x - mu) ** 2) / (2 * sigma ** 2)
        return coefficent * math.exp(exponent)

    def _calc_conditional_probabilities(self, X, y):
        conditional_probabilities = defaultdict(list)

        for c in self.classes:
            X_c = X[y == c]

            for term in X_c.T:
                gauss = Gaussian(np.mean(term), np.std(term))
                conditional_probabilities[c].append(gauss)

        return conditional_probabilities

    def _calc_p_B(self, X):
        predictor_priors = []

        for term in X.T:
            gauss = Gaussian(np.mean(term), np.std(term))
            predictor_priors.append(gauss)

        return predictor_priors

    def fit(self, X, y):
        self.priors = self._calc_priors(y)
        self.conditional_probabilities = self._calc_conditional_probabilities(X, y)
        self.predictor_priors = self._calc_p_B(X)


    def predict_proba_row(self, x):
        probs = {}

        for cls in self.classes:
            log_p_A = math.log(self.priors[cls])

            log_p_B_A = 0
            log_p_B = 0
            for i, v in enumerate(x):
                p_B_A = self.conditional_probabilities[cls][i].calc(v)
                p_B = self.predictor_priors[i].calc(v)
                log_p_B_A += math.log(p_B_A)
                log_p_B += math.log(p_B)

            probs[cls] = (log_p_B_A + log_p_A) - log_p_B

        return probs
