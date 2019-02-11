import math
from typing import List, Tuple, Dict, Any, Union

import pandas as pd
import logging

"""
Questions:

Entropy
=======
How should entropy be normalised? 
The max value is 1, you sum them for each class.
https://www.saedsayad.com/decision_tree.htm

The implementations seem to use -p log2 p, without q, that doesn't make a symmetric distribution
It changes the point of inflection as well from 0.5 to ~0.35
without q is used for multi-class problems, still don't get the movement of the mid-point
the idea being for each class you add on another -p log2 p, just like the q example is for th negative class

This had a good explanation of what is going on with the information gain calculation
machine learning (1997).pdf
page 70

Todos
=====
tests for max depth, min_samples_*
min_impurity_reduction for regression
creating a base tree class
calculate_gini and entropy should be updated to work with pd.Series as input
start score should be None rather than arbitrarily small number
"""


class GNSDecisionTreeClassifier:
    def __init__(
            self,
            X_cols: List[str],
            y_col: str,
            criterion: str = 'gini',
            max_depth: int = 10,
            min_samples_leaf: int = 1,
            min_samples_split: int = 2,
            min_impurity_reduction: float = 1e-6):

        self.X_cols = X_cols
        self.y_col = y_col
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_reduction = min_impurity_reduction
        self.tree = None
        self.y_values = None

        if criterion == "gini":
            self.criterion = self._calculate_gini_gain
        # elif criterion == "entropy":
        #     self.criterion = self._calculate_entropy
        else:
            raise ValueError("Unknown criterion '{}' passed".format(criterion))

    @staticmethod
    def _calculate_gini(left_branch: Dict[int, int], right_branch: Dict[int, int]) -> float:
        """Pass dictionary of counts"""

        def calc_branch_gini(branch):
            total = sum(branch.values())
            impurity = 0
            for v, count in branch.items():
                p = (count / total)
                impurity += p * (1 - p)
            return impurity, total

        left_impurity, left_n = calc_branch_gini(left_branch)
        right_impurity, right_n = calc_branch_gini(right_branch)
        n = left_n + right_n

        gini = (left_impurity * left_n / n) + (right_impurity * right_n / n)
        return gini

    @staticmethod
    def _calculate_entropy(left_branch: Dict[int, int], right_branch: Dict[int, int]) -> float:
        def calc_branch_entropy(branch):
            total = sum(branch.values())
            n_classes = len(branch)
            entropy = 0
            for v, count in branch.items():
                if count > 0:
                    p = (count / total)
                    q = 1 - p
                    if q != 0:
                        # this stops log(0), and as q goes to 0 the entropy becomes 0
                        entropy += (- p * math.log2(p) - q * math.log2(q)) / n_classes
            return entropy, total

        left_entropy, left_n = calc_branch_entropy(left_branch)
        right_entropy, right_n = calc_branch_entropy(right_branch)
        n = left_n + right_n

        entropy = (left_entropy * left_n / n) + (right_entropy * right_n / n)
        return entropy

    # @staticmethod
    # def _calculate_information_gain(left_branch: Dict[int, int], right_branch: Dict[int, int]) -> float:
    #     # key is class, value is count
    #     overall_class_counts = {}
    #     total = 0
    #     for klass in left_branch.keys():
    #         v = left_branch[klass]
    #         v += right_branch.get(klass, 0)
    #         overall_class_counts[klass] = v
    #         total += v

    @staticmethod
    def _calculate_gini_gain(left_branch: pd.Series, right_branch: pd.Series) -> float:
        left_branch = dict(left_branch.value_counts())
        right_branch = dict(right_branch.value_counts())
        return 1 - GNSDecisionTreeClassifier._calculate_gini(left_branch, right_branch)

    def _calculate_stump_split(self, fvs):
        max_feature = None
        max_split_score = -1e10
        max_split_value = None

        for feature in self.X_cols:
            unique_feature = fvs[feature].unique()
            unique_feature.sort()

            for split_i in range(len(unique_feature) - 1):
                # get the mid point of this value and the next
                split_v = unique_feature[split_i] + ((unique_feature[split_i + 1] - unique_feature[split_i]) / 2)

                mask = fvs[feature] > split_v
                left = fvs[mask][self.y_col]
                right = fvs[~mask][self.y_col]

                gain = self.criterion(left, right)

                if gain > max_split_score:
                    max_split_score = gain
                    max_split_value = split_v
                    max_feature = feature

        return max_split_value, max_feature, max_split_score

    def _get_leaf_counts(self, fvs: pd.DataFrame):
        return dict(fvs[self.y_col].value_counts())

    def _fit(
            self,
            fvs: pd.DataFrame,
            tree: Dict[str, Any],
            splits: List[Tuple[float, str]],
            level: int = 0,
            split_score: float = -1e6) -> Union[dict, Tuple[Dict[str, Any], Tuple[float, str]]]:

        # stopping criteria
        if fvs.shape[0] <= self.min_samples_split:
            return self._get_leaf_counts(fvs)
        elif fvs.shape[0] <= self.min_samples_leaf:
            return self._get_leaf_counts(fvs)
        elif fvs[self.y_col].unique().size == 1:
            return self._get_leaf_counts(fvs)
        elif level > self.max_depth:
            return self._get_leaf_counts(fvs)

        split_value, split_feature, new_split_score = self._calculate_stump_split(fvs)
        if self.min_impurity_reduction > (new_split_score - split_score):
            return self._get_leaf_counts(fvs)

        splits.append((split_value, split_feature))
        mask = fvs[split_feature] < split_value

        left = fvs[mask]
        right = fvs[~mask]

        common_logging_data = level, fvs.shape[0], dict(fvs[self.y_col].value_counts()), new_split_score, split_score
        common_args = {
            "level": level,
            "split_score": new_split_score,
            "splits": splits,
            "tree": tree
        }

        logging.debug("left\t", *common_logging_data)
        left = self._fit(left, **common_args)

        logging.debug("right\t", *common_logging_data)
        right = self._fit(right, **common_args)

        return {'left': left, 'right': right}, (split_value, split_feature)

    @staticmethod
    def _predict_one(fv: pd.Series, tree: Dict):
        if isinstance(tree, dict):
            return tree

        split_value, split_feature = tree[1]
        if fv[split_feature] < split_value:
            return GNSDecisionTreeClassifier._predict_one(fv, tree[0]['left'])
        else:
            return GNSDecisionTreeClassifier._predict_one(fv, tree[0]['right'])

    def fit(self, fvs: pd.DataFrame):
        self.tree = self._fit(
            fvs=fvs,
            tree={},
            splits=[]
        )
        return self

    def predict_counts(self, fvs: pd.DataFrame):
        return fvs.apply(lambda fv: self._predict_one(fv, self.tree), axis=1)

    def predict_probability(self, fvs: pd.DataFrame):
        preds = self.predict_counts(fvs)
        for d in preds:
            d.keys()
        pass

    def predict(self, fvs: pd.DataFrame):
        preds = self.predict_counts(fvs)
        return preds.map(lambda d: max(d, key=d.get))


class GNSDecisionTreeRegressor(GNSDecisionTreeClassifier):
    def __init__(
            self,
            X_cols: List[str],
            y_col: str,
            criterion: str = 'mse',
            max_depth: int = 10,
            min_samples_leaf: int = 1,
            min_samples_split: int = 2):

        super().__init__(
            X_cols=X_cols,
            y_col=y_col,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_reduction=0  # TODO check this
        )
        if criterion == "mse":
            self.criterion = self._mse_score
        else:
            raise ValueError("Unknown criterion '{}' passed".format(criterion))

    @staticmethod
    def _mse(left_branch: pd.Series, right_branch: pd.Series) -> float:
        output = 0
        n = left_branch.shape[0] + right_branch.shape[0]

        for br in [left_branch, right_branch]:
            mean = br.mean()
            mse = ((br - mean)**2).sum()
            output += mse * (br.shape[0] / n)

        return output

    @staticmethod
    def _mse_score(left_branch: pd.Series, right_branch: pd.Series) -> float:
        return -1 * GNSDecisionTreeRegressor._mse(left_branch, right_branch)
