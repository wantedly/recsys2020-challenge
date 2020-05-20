import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List


def label_encoding(col: str, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray]:
    le = LabelEncoder()
    train_label = list(train[col].astype(str).values)
    test_label = list(test[col].astype(str).values)
    total_label = train_label + test_label
    le.fit(total_label)
    train_feature = le.transform(train_label)
    test_feature = le.transform(test_label)

    return train_feature, test_feature


def count_encoding(col: str, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
    total = pd.concat([train, test], ignore_index=True, sort=False)
    count_map = total[col].value_counts().to_dict()
    train_feature = train[col].map(count_map)
    test_feature = test[col].map(count_map)

    return train_feature, test_feature


def target_encoding(col: str, train: pd.DataFrame, test: pd.DataFrame,
                    target: str, folds_ids: List[Tuple[np.asarray]]):
    train[f"{col}_ta"] = np.nan
    for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
        target_mean = train.iloc[trn_idx].groupby(col)[target].mean()
        train.set_index(col, inplace=True)
        train.iloc[val_idx, -1] = target_mean
        train.reset_index(inplace=True)

    test_target_mean = train.groupby(col)[target].mean()
    test[f"{col}_ta"] = np.nan
    test.set_index(col, inplace=True)
    test.iloc[:,-1] = test_target_mean
    test.reset_index(inplace=True)

    return train[f"{col}_ta"], test[f"{col}_ta"]


def target_encoding_lower_limit(col: str, train: pd.DataFrame, test: pd.DataFrame,
                    target: str, folds_ids: List[Tuple[np.asarray]]):
    from scipy import stats

    alpha = 0.90
    func_lower_limit = lambda x: stats.binom.interval(alpha=alpha, n=x["size"], p=x["mean"], loc=0)[0] / x["size"]

    train[f"{col}_ta"] = np.nan
    for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
        target_mean = train.iloc[trn_idx].groupby(col)[target].mean().rename("mean")
        target_size = train.iloc[trn_idx].groupby(col)[target].size().rename("size")
        target_info = pd.concat([target_mean, target_size], axis=1)
        target_info[target] = target_info.apply(func_lower_limit, axis=1)
        target_info.loc[target_info["mean"]==0, target] = 0   # scipy bug https://github.com/scipy/scipy/issues/11026

        train.set_index(col, inplace=True)
        train.iloc[val_idx, -1] = target_info[target]
        train.reset_index(inplace=True)

    test_target_mean = train.groupby(col)[target].mean().rename("mean")
    test_target_size = train.groupby(col)[target].size().rename("size")
    test_target_info = pd.concat([test_target_mean, test_target_size], axis=1)
    test_target_info[target] = test_target_info.apply(func_lower_limit, axis=1)
    test_target_info.loc[test_target_info["mean"]==0, target] = 0   # scipy bug https://github.com/scipy/scipy/issues/11026

    test[f"{col}_ta"] = np.nan
    test.set_index(col, inplace=True)
    test.iloc[:,-1] = test_target_info[target]
    test.reset_index(inplace=True)

    return train[f"{col}_ta"], test[f"{col}_ta"]
