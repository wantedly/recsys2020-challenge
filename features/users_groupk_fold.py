import numpy as np
import pandas as pd
from base import BaseFeature
from utils import reduce_mem_usage
from typing import Optional, List, Tuple
from sklearn.model_selection import KFold
import tempfile
import os


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"

FOLD = 3
RANDOM_STATE = 71
SHUFFLE = True

class UsersGroupKFold(BaseFeature):
    def import_columns(self):
        return [
            "engaging_user_id",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        group_name = "engaging_user_id"
        group = df_train_input[group_name]
        unique_group = group.unique()

        kf = KFold(
            n_splits=FOLD,
            shuffle=SHUFFLE,
            random_state=RANDOM_STATE
        )
        split = kf.split(unique_group)

        val_position = np.zeros(len(df_train_input)).astype(np.int8)
        for i_fold, (trn_group_idx, val_group_idx) in enumerate(split):
            trn_group = unique_group[trn_group_idx]
            val_group = unique_group[val_group_idx]
            is_trn = group.isin(trn_group)
            is_val = group.isin(val_group)
            trn_idx = df_train_input[is_trn].index
            val_idx = df_train_input[is_val].index
            val_position[val_idx] = i_fold

            df_train_features["val_position"] = val_position
            df_train_input["val_position"] = val_position   # debug

        print(df_train_features.shape)
        print(df_train_features["val_position"].value_counts())
        print(df_train_features.isnull().sum())

        print(df_train_input[group_name].nunique())
        print(df_train_input.groupby(["val_position"]).apply(lambda x: len(np.unique(x[group_name]))))

        return df_train_features, df_test_features

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        self._logger.info(f"Running with debugging={self.debugging}")
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            if TESTING:
                test_path = os.path.join(tempdir, f"{self.name}_test.ftr")
                test_table = f"`{PROJECT_ID}.recsys2020.test`"
            else:
                test_path = os.path.join(tempdir, f"{self.name}_val.ftr")
                test_table = f"`{PROJECT_ID}.recsys2020.val`"
            train_path = os.path.join(tempdir, f"{self.name}_training.ftr")
            train_table = f"`{PROJECT_ID}.recsys2020.training`"
            self.read_and_save_features(
                train_table, test_table, train_path, test_path,
            )
            self._upload_to_gs([train_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_output_path: str,
    ) -> None:
        df_train_input = self._read_from_bigquery(train_table_name)
        df_test_input = self._read_from_bigquery(test_table_name)
        df_train_features, df_test_features = self.make_features(
            df_train_input, df_test_input
        )
        assert (
            df_train_input.shape[0] == df_train_features.shape[0]
        ), "generated train features is not compatible with the table"

        df_train_features.columns = f"{self.name}_" + df_train_features.columns

        if self.save_memory:
            self._logger.info("Reduce memory size - train data")
            df_train_features = reduce_mem_usage(df_train_features)

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)


if __name__ == "__main__":
    UsersGroupKFold.main()
