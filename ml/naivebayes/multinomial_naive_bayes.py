from collections import defaultdict

import math
import numpy as np

from ml.naivebayes.base_naive_bayes import CountsNaiveBayes


class MultinomialNaiveBayes(CountsNaiveBayes):
    def __init__(self, sm_factor: float = 1):
        super().__init__()
        self.sm_factor = sm_factor
        self.conditional_probabilities = defaultdict(list)
        self.log_p_B = None

    def _calc_conditional_probabilities(self, X, y):
        conditional_probabilities = defaultdict(list)
        n_terms = X.shape[1]

        for c in self.priors.keys():
            X_c = X[y == c]
            class_total_counts = X_c.sum()

            for term in X_c.T:
                numerator = term.sum() + 1
                denominator = class_total_counts + n_terms
                conditional_probabilities[c].append(numerator / denominator)

        return conditional_probabilities

    def _calc_p_B(self, X):
        log_p_B = 0
        n_terms = X.shape[1]
        total_counts = X.sum()

        for term in X.T:
            numerator = term.sum() + 1
            denominator = total_counts + n_terms
            log_p_B += math.log(numerator / denominator)

        return math.exp(log_p_B)

    def fit(self, X, y):
        """Each feature/column is a count of how often that feature occurs in the training example"""
        self.priors = self._calc_priors(y)

        # calculate conditional probabilities for each term
        self.conditional_probabilities = self._calc_conditional_probabilities(X, y)

        # calculate probability of the data
        self.predictor_prior = self._calc_p_B(X)

        return self

    def _predict_proba_row(self, x):
        probs = np.full(len(self.priors), fill_value=0, dtype=np.float)
        for i, cls in enumerate(self.priors.keys()):
            log_prob = math.log(self.priors[cls])

            log_prob += sum([
                math.log(p) * n
                for p, n in zip(self.conditional_probabilities[cls], x)
                if n > 0
            ])

            # TODO - something wrong with p_B
            # probs[i] = math.exp(log_prob - self.log_p_B)
            probs[i] = math.exp(log_prob)
        return probs

    def predict_proba(self, X):
        return [self._predict_proba_row(row) for row in X]

    def predict_proba_with_p_B(self, X):
        # TODO - to get true probabilities I need to add in the divide by p_B
        pass

    def predict(self, X):
        return [np.argmax(cls_probs) for cls_probs in self.predict_proba(X)]
