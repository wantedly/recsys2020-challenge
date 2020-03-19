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
    def __init__(self, name: Optional[str], save_memory: bool = True) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__
        self.save_memory = save_memory
        self._logger = Logger(self.__class__.__name__)

    @abc.abstractproperty
    def import_columns(self) -> List[str]:
        """この特徴量を作るのに必要なカラムを指定する
        """
        ...

    @abc.abstractmethod
    def make_features(
        self, df_train_input: pd.DataFrame, df_val_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """BigQuery から取得した生データの DataFrame を特徴量に変換する
        """
        ...

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            if TESTING:
                valid_path = os.path.join(tempdir.name, f"{self.name}_test.ftr")
                valid_table = "`wantedly-individual-naomichi.recsys2020.test`"
            else:
                valid_path = os.path.join(tempdir.name, f"{self.name}_val.ftr")
                valid_table = "`wantedly-individual-naomichi.recsys2020.val`"
            train_path = os.path.join(tempdir.name, f"{self.name}_training.ftr")
            train_table = "`wantedly-individual-naomichi.recsys2020.training`"
            self.read_and_save_features(
                train_table, valid_table, train_path, valid_path,
            )
            self._upload_to_gs([valid_path, train_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        valid_table_name: str,
        train_output_path: str,
        valid_table_name: str,
    ) -> None:
        df_train_input = self._read_from_bigquery(train_table_name)
        df_valid_input = self._read_from_bigquery(valid_table_name)
        df_train_features, df_valid_features = self.make_features(
            df_train_input, df_valid_input
        )
        assert (
            df_train_input.shape[0] == df_train_features.shape[0]
        ), "generated train features is not compatible with the table"
        assert (
            df_valid_input.shape[0] == df_valid_features.shape[0]
        ), "generated valid features is not compatible with the table"
        self._logger.info(f"Saving features into {output_path}")
        df_train_features.columns = f"{self.name}_" + df_train_features.columns
        df_valid_features.columns = f"{self.name}_" + df_valid_features.columns
        if self.save_memory:
            # TODO: fp16 にしたりする。
            df_train_features = df_train_features.convert_dtypes()
            df_valid_features = df_valid_features.convert_dtypes()
        df_train_features.to_feather(train_output_path)
        df_valid_features.to_feather(valid_output_path)

    def _read_from_bigquery(self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = """
            select {}
            from {}
            order by tweet_id, engaging_user_id
        """.format(
            self.import_columns.join(", "), table_name
        )
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
