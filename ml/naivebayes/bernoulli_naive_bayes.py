from collections import defaultdict
from typing import Dict, Any

import math

from ml.naivebayes.base_naive_bayes import CountsNaiveBayes


class BernoulliNaiveBayes(CountsNaiveBayes):
    def __init__(self, use_p_B: bool = False):
        super().__init__()
        self.use_p_B = use_p_B

    def fit(self, X, y):
        """Each feature is a boolean for the presence of the feature in the example"""
        X = self._pandas_to_np(X)

        self.priors = self._calc_priors(y)
        self.conditional_probabilities = self._calc_conditional_probabilities(X, y)
        self.predictor_prior = self._calc_p_B(X)
        return self

    def _calc_conditional_probabilities(self, X, y):
        conditional_probabilities = defaultdict(list)
        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)

            for feature in X_c.T:
                n_ct = feature.sum()
                p_B_A = (n_ct + 1) / (n_c + 2)
                conditional_probabilities[c].append(p_B_A)

        return conditional_probabilities

    def _calc_p_B(self, X):
        pass

    def _predict_proba_row(self, x) -> Dict[Any, float]:
        probs = {}
        for cls in self.classes:
            log_prob = math.log(self.priors[cls])
            for j, term in enumerate(x):
                con_prob = self.conditional_probabilities[cls][j]
                if term:
                    log_prob += math.log(con_prob)
                else:
                    log_prob += math.log(1 - con_prob)
            probs[cls] = math.exp(log_prob)

        if self.use_p_B:
            p_B = sum(probs.values())
            return {k: v / p_B for k, v in probs.items()}
        else:
            return probs
