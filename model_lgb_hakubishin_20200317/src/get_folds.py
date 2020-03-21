import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold


class Fold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=71):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_kfold(self, train):
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        folds_ids = []
        for trn_idx, val_idx in kf.split(train):
            folds_ids.append((
                pd.Int64Index(trn_idx), pd.Int64Index(val_idx)
            ))

        return folds_ids

    def get_stratifiedkfold(self, train, y_train):
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        folds_ids = []
        for trn_idx, val_idx in kf.split(train, y_train):
            folds_ids.append((
                pd.Int64Index(trn_idx), pd.Int64Index(val_idx)
            ))
        return folds_ids

    def get_groupkfold(self, train, group_name):
        group = train[group_name]
        unique_group = group.unique()

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        folds_ids = []
        for trn_group_idx, val_group_idx in kf.split(unique_group):
            trn_group = unique_group[trn_group_idx]
            val_group = unique_group[val_group_idx]
            is_trn = group.isin(trn_group)
            is_val = group.isin(val_group)
            trn_idx = train[is_trn].index
            val_idx = train[is_val].index
            folds_ids.append((trn_idx, val_idx))

        return folds_ids

    def get_timeseriesfold(self, train, time_name, val_period_list, train_future=False):
        """
        Example:
            - if train_future is False
            fold1: trn, val
            fold2: trn, trn, val
            fold3: trn, trn, trn, val
            - if train_future is True
            fold1: trn, val, trn, trn
            fold2: trn, trn, val, trn
            fold3: trn, trn, trn, val
        """
        folds_ids = []
        for val_period in val_period_list:
            if train_future:
                is_trn = train[time_name] != val_period
                is_val = train[time_name] == val_period
            else:
                is_trn = train[time_name] < val_period
                is_val = train[time_name] == val_period

            trn_idx = train[is_trn].index
            val_idx = train[is_val].index
            folds_ids.append((trn_idx, val_idx))

        return folds_ids

    def get_holdout(self, train, test_ratio=0.3):
        train_index = train.index
        train_size = int(len(train) * (1 - test_ratio))

        if self.shuffle:
            np.random.seed(self.random_state)
            train_index = np.random.permutation(train_index)

        folds_ids = []
        trn_idx = train_index[:train_size]
        val_idx = train_index[train_size:]
        folds_ids.append((trn_idx, val_idx))

        return folds_ids

    def get_timeseries_holdout(self, train, time_name, val_threshold):
        folds_ids = []
        is_trn = train[time_name] < val_threshold
        is_val = train[time_name] >= val_threshold
        trn_idx = train[is_trn].index
        val_idx = train[is_val].index
        folds_ids.append((trn_idx, val_idx))

        return folds_ids
