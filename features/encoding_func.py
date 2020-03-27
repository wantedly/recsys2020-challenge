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
