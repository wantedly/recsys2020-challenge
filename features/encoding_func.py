import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


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
