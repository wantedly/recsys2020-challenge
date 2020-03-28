import pandas as pd
from base import BaseFeature
from utils import reduce_mem_usage
from typing import Optional, List, Tuple
import tempfile
import os


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class KeyCategories(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        for col in self.import_columns():
            df_test_features[col] = df_test_input[col].values

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
            self._upload_to_gs([test_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_output_path: str,
    ) -> None:
        df_train_input = pd.DataFrame()   # dummy
        df_test_input = self._read_from_bigquery(test_table_name)
        df_train_features, df_test_features = self.make_features(
            df_train_input, df_test_input
        )
        assert (
            df_test_input.shape[0] == df_test_features.shape[0]
        ), "generated test features is not compatible with the table"

        df_test_features.columns = f"{self.name}_" + df_test_features.columns

        self._logger.info(f"Saving features to {test_output_path}")
        df_test_features.to_feather(test_output_path)


if __name__ == "__main__":
    KeyCategories.main()
