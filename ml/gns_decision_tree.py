from typing import List, Tuple, Dict, Any, Union

import pandas as pd
import unittest
import logging


class GNSDecisionTreeClassifier:
    def __init__(
            self,
            X_cols: List[str],
            y_col: str,
            criterion: str = 'gini',
            max_depth: int = 10,
            min_impurity_reduction: float = 1e-6):

        self.X_cols = X_cols
        self.y_col = y_col
        self.min_impurity_reduction = min_impurity_reduction
        self.max_depth = max_depth
        self.tree = None

        if criterion == "gini":
            self.criterion = self._calculate_gini
        else:
            raise ValueError("Unknown criterion '{}' passed".format(criterion))

    def _calculate_gini(self, left_branch, right_branch):
        def calc_branch_gini(branch):
            counts = dict(branch[self.y_col].value_counts())
            total = sum(counts.values())
            output = 0
            for v, count in counts.items():
                output += (count / total) ** 2
            return output, total

        left_gini, left_total = calc_branch_gini(left_branch)
        right_gini, right_total = calc_branch_gini(right_branch)
        total = left_total + right_total

        gini = (left_gini * left_total / total) + (right_gini * right_total / total)
        return gini

    def _calculate_stump_split(self, fvs):
        splits = {}

        for feature in self.X_cols:
            splits[feature] = []
            unique_feature = fvs[feature].unique()
            unique_feature.sort()

            for split_i in range(len(unique_feature)):
                split_v = unique_feature[split_i]  # TODO this should be the mid point

                mask = fvs[feature] > split_v
                left = fvs[mask]
                right = fvs[~mask]

                gini = self._calculate_gini(left, right)
                splits[feature].append((gini, split_v))

        max_feature = ""
        max_split_score = 0
        max_split_value = 0

        for feature in self.X_cols:
            split_score, split_value = max(splits[feature])

            if split_score > max_split_score:
                max_split_score = split_score
                max_split_value = split_value
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
            split_score: float = 0) -> Union[dict, Tuple[Dict[str, Any], Tuple[float, str]]]:

        if fvs.shape[0] < 2:
            return self._get_leaf_counts(fvs)
        elif fvs[self.y_col].unique().size == 1:
            return self._get_leaf_counts(fvs)
        elif level == self.max_depth:
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
        pass

    def predict(self, fvs: pd.DataFrame):
        preds = self.predict_counts(fvs)
        return preds.map(lambda d: max(d, key=d.get))


if __name__ == "__main__":
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    # Data setup
    df = sns.load_dataset("iris")
    df = df[df.species.isin(["versicolor", "virginica"])].copy()
    df["species_i"] = (df["species"] == "versicolor").astype(np.uint8)

    train, test = train_test_split(df, random_state=44)
    X_cols = train.columns[:-2].tolist()
    y_col = "species_i"

    # check model
    dt = GNSDecisionTreeClassifier(X_cols, y_col)
    dt = dt.fit(train)
    accuracy = (test[y_col] == dt.predict(test)).mean()
    print("Accuracy: {:.4}".format(accuracy))
    assert accuracy > .85
