import argparse
import abc
from typing import Optional, List, Tuple
from logging import Logger
import tempfile
import os

import pandas as pd
from google.cloud import storage


TESTING = False
GCS_BUCKET_NAME = "gs://recsys2020-challenge-wantedly"


class BaseFeature(abc.ABC):
    save_memory: bool = True

    def __init__(self, debugging: bool = False, **kwargs) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.debugging = debugging
        self._logger = Logger(self.__class__.__name__)

    @abc.abstractproperty
    def import_columns(self) -> List[str]:
        """この特徴量を作るのに必要なカラムを指定する
        """
        ...

    @abc.abstractmethod
    def make_features(
        self, df_train_input: pd.DataFrame, df_test_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """BigQuery から取得した生データの DataFrame を特徴量に変換する
        """
        ...

    @abc.abstractclassmethod
    def add_feature_specific_arguments(cls, parser: argparse.ArgumentParser):
        return

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        cls.add_feature_specific_arguments(parser)
        args = parser.parse_args()
        instance = cls(debugging=args.debugging, **vars(args))
        instance.run()

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        self._logger.info(f"Running with debugging={self.debugging}")
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            if TESTING:
                test_path = os.path.join(tempdir.name, f"{self.name}_test.ftr")
                test_table = "`wantedly-individual-naomichi.recsys2020.test`"
            else:
                test_path = os.path.join(tempdir.name, f"{self.name}_val.ftr")
                test_table = "`wantedly-individual-naomichi.recsys2020.val`"
            train_path = os.path.join(tempdir.name, f"{self.name}_training.ftr")
            train_table = "`wantedly-individual-naomichi.recsys2020.training`"
            self.read_and_save_features(
                train_table, test_table, train_path, test_path,
            )
            is not self.debugging:
                self._upload_to_gs([test_path, train_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_table_name: str,
    ) -> None:
        df_train_input = self._read_from_bigquery(train_table_name)
        df_test_input = self._read_from_bigquery(test_table_name)
        df_train_features, df_test_features = self.make_features(
            df_train_input, df_test_input
        )
        assert (
            df_train_input.shape[0] == df_train_features.shape[0]
        ), "generated train features is not compatible with the table"
        assert (
            df_test_input.shape[0] == df_test_features.shape[0]
        ), "generated test features is not compatible with the table"
        self._logger.info(f"Saving features into {output_path}")
        df_train_features.columns = f"{self.name}_" + df_train_features.columns
        df_test_features.columns = f"{self.name}_" + df_test_features.columns
        if self.save_memory:
            # TODO: fp16 にしたりする。
            df_train_features = df_train_features.convert_dtypes()
            df_test_features = df_test_features.convert_dtypes()
        df_train_features.to_feather(train_output_path)
        df_test_features.to_feather(test_output_path)

    def _read_from_bigquery(self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = """
            select {}
            from {}
            order by tweet_id, engaging_user_id
        """.format(
            self.import_columns.join(", "), table_name
        )
        if self.debugging:
            query += " limit 10000"
        return pd.read_gbq(
            query, dialect="standard", project_id="wantedly-individual-naomichi"
        )

    def _upload_to_gs(self, files: List[str]):
        client = storage.Client(project="wantedly-individual-naomichi")
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        for filename in files:
            basename = os.path.basename(filename)
            blob = bucket.get_blob(os.path.join("features", basename))
            self._logger.info(f"Uploading {basename} to {blob.path}")
            blob.upload_from_filename(filename)
