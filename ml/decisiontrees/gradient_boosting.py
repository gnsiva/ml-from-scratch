import logging
from typing import Dict, Optional, List, Any, Tuple, Union

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


class GradientBoostingRegressor:
    def __init__(
            self,
            learning_rate: float = 0.1,
            criterion: str = "mse",
            subsample_fraction: float = 1.0,
            max_features: float = 1.0,
            n_estimators: int = 10,
            min_score_improvement: float = 1e-4,
            tree_params: Optional[Dict] = None):

        self.learning_rate = learning_rate
        self.criterion = criterion
        self.subsample_fraction = subsample_fraction
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_score_improvement = min_score_improvement

        self.tree_params = {}
        if tree_params:
            self.tree_params = tree_params

        # internal parameters
        self.trees = []
        self.y0 = None

    def fit(self, X, y):
        self.trees = []

        self.y0 = y.mean()
        yhat = np.full(X.shape[0], self.y0)
        residuals = y - yhat
        mse = (residuals ** 2).sum()

        for i in range(self.n_estimators):
            # fit a tree on the residuals
            dt = DecisionTreeRegressor(**self.tree_params)
            dt = dt.fit(X, residuals)

            # update the total prediction
            yhat += self.learning_rate * dt.predict(X)
            residuals = y - yhat

            # check stopping criteria
            new_mse = (residuals ** 2).sum()
            if (mse - new_mse) < self.min_score_improvement:
                logging.debug("Reached minimum score improvement, exiting on estimator {}".format(i + 1))
                break

            mse = new_mse
            self.trees.append(dt)

        return self

    def predict(self, X):
        yhat = np.full(X.shape[0], self.y0)
        for tree in self.trees:
            yhat += self.learning_rate * tree.predict(X)

        return yhat


class GradientBoostingMAERegressor:
    def __init__(
            self,
            X_cols: List[str],
            y_col: str,
            learning_rate: float = 0.1,
            criterion: str = "mae",
            subsample_fraction: float = 1.0,
            max_features: float = 1.0,
            n_estimators: int = 10,
            min_score_improvement: float = 1e-4,
            tree_params: Optional[Dict] = None):

        self.X_cols = X_cols
        self.y_col = y_col
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.subsample_fraction = subsample_fraction
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_score_improvement = min_score_improvement

        self.tree_params = {}
        if tree_params:
            self.tree_params = tree_params

        # internal parameters
        self.trees = []
        self.y0 = None

    def calculate_sign_residuals(self, y, yhat):
        residuals = y - yhat
        zero_mask = np.isclose(residuals, 0)

        residuals[residuals > 0] = 1
        residuals[residuals < 0] = -1
        residuals[zero_mask] = 0
        return residuals

    def fit(self, X: pd.DataFrame):
        self.trees = []

        y = X[self.y_col]
        self.y0 = np.median(y)
        yhat = np.full(X.shape[0], self.y0)
        X["residuals"] = self.calculate_sign_residuals(y, yhat)

        mae = mean_absolute_error(y, yhat)

        for i in range(self.n_estimators):
            # fit a tree on the residuals
            dt = _GBRDecisionTreeRegressor(
                X_cols=self.X_cols,
                residual_y_col="residuals",
                actual_y_col=self.y_col,
                **self.tree_params)
            dt = dt.fit(X)

            # update the total prediction
            yhat += self.learning_rate * dt.predict_median_leaf(X, yhat)
            X["residuals"] = self.calculate_sign_residuals(y, yhat)

            # check stopping criteria
            new_mae = mean_absolute_error(y, yhat)
            if (mae - new_mae) < self.min_score_improvement:
                logging.debug("Reached minimum score improvement, exiting on estimator {}".format(i + 1))
                break

            mae = new_mae
            self.trees.append(dt)

        return self

    def predict(self, X):
        yhat = np.full(X.shape[0], self.y0)
        for tree in self.trees:
            yhat += self.learning_rate * tree.predict(X)

        return yhat


class _FinalData:
    def __init__(self, tree: Dict, leaf_values: float):
        self.tree = tree
        self.leaf_values = leaf_values


class _GBRDecisionTreeRegressor:
    def __init__(
            self,
            X_cols: List[str],
            residual_y_col: str,
            actual_y_col: Optional[str] = None,
            criterion: str = "mse",
            max_depth: int = 10,
            min_samples_leaf: int = 1,
            min_samples_split: int = 2,
            min_impurity_reduction: float = 1e-6):

        self.X_cols = X_cols
        self.residual_y_col = residual_y_col
        self.actual_y_col = actual_y_col
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_reduction = min_impurity_reduction
        self.tree = None

        self.criterion = self._get_criterion_function(criterion)

    def _get_criterion_function(self, criterion: str):
        if criterion == "mse":
            return self._mse_score
        else:
            raise ValueError("Unknown criterion '{}' passed".format(criterion))

    def _calculate_stump_split(self, fvs):
        max_feature = None
        max_split_score = None
        max_split_value = None

        for feature in self.X_cols:
            unique_feature = fvs[feature].unique()
            unique_feature.sort()

            for split_i in range(len(unique_feature) - 1):
                # get the mid point of this value and the next
                split_v = unique_feature[split_i] + ((unique_feature[split_i + 1] - unique_feature[split_i]) / 2)

                mask = fvs[feature] > split_v
                left = fvs[mask][self.residual_y_col]
                right = fvs[~mask][self.residual_y_col]

                gain = self.criterion(left, right)

                if max_split_score is None or gain > max_split_score:
                    max_split_score = gain
                    max_split_value = split_v
                    max_feature = feature

        return max_split_value, max_feature, max_split_score

    def _get_leaf_counts(self, fvs: pd.DataFrame) -> _FinalData:
        leaf_values = None
        if self.actual_y_col:
            leaf_values = fvs[self.actual_y_col].copy()
        return _FinalData(dict(fvs[self.residual_y_col].value_counts()), leaf_values)

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
        return -1 * _GBRDecisionTreeRegressor._mse(left_branch, right_branch)

    def _fit(
            self,
            fvs: pd.DataFrame,
            tree: Dict[str, Any],
            splits: List[Tuple[float, str]],
            level: int = 0,
            split_score: Optional[float] = None) -> Union[_FinalData, Tuple[Dict[str, Any], Tuple[float, str]]]:

        # stopping criteria
        if fvs.shape[0] <= self.min_samples_split or \
                fvs.shape[0] <= self.min_samples_leaf or \
                fvs[self.residual_y_col].unique().size == 1 or \
                level > self.max_depth:
            return self._get_leaf_counts(fvs)

        split_value, split_feature, new_split_score = self._calculate_stump_split(fvs)
        if split_score is not None:
            improvement = new_split_score - split_score
            if self.min_impurity_reduction > improvement:
                return self._get_leaf_counts(fvs)

        splits.append((split_value, split_feature))
        mask = fvs[split_feature] < split_value

        left = fvs[mask]
        right = fvs[~mask]

        common_logging_data = level, fvs.shape[0], dict(fvs[self.residual_y_col].value_counts()), new_split_score, split_score
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
        if isinstance(tree, _FinalData):
            return tree.tree

        split_value, split_feature = tree[1]
        if fv[split_feature] < split_value:
            return _GBRDecisionTreeRegressor._predict_one(fv, tree[0]['left'])
        else:
            return _GBRDecisionTreeRegressor._predict_one(fv, tree[0]['right'])

    def predict_counts(self, fvs: pd.DataFrame):
        return fvs.apply(lambda fv: self._predict_one(fv, self.tree), axis=1)

    def predict(self, fvs: pd.DataFrame) -> pd.Series:
        preds = self.predict_counts(fvs)

        def calculate_average(d: Dict[float, int]):
            total = sum(d.values())
            return sum([k * (v / total) for k, v in d.items()])

        return preds.map(calculate_average)

    @staticmethod
    def _predict_for_mae(fv: pd.Series, tree: Union[_FinalData, Tuple]):
        if isinstance(tree, _FinalData):
            return np.median(fv["yhat"] - tree.leaf_values)

        split_value, split_feature = tree[1]
        if fv[split_feature] < split_value:
            return _GBRDecisionTreeRegressor._predict_for_mae(fv, tree[0]['left'])
        else:
            return _GBRDecisionTreeRegressor._predict_for_mae(fv, tree[0]['right'])

    def predict_median_leaf(self, fvs: pd.DataFrame, yhat: pd.Series) -> pd.Series:
        fvs["yhat"] = yhat
        return fvs.apply(lambda fv: self._predict_for_mae(fv, self.tree), axis=1)

    def fit(self, fvs: pd.DataFrame):
        self.tree = self._fit(
            fvs=fvs,
            tree={},
            splits=[]
        )
        return self
