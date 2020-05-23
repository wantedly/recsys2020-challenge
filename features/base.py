import argparse
import abc
from typing import Optional, List, Tuple
from logging import Logger, StreamHandler, INFO, Formatter
import tempfile
import os

import pandas as pd
from google.cloud import storage, bigquery
from utils import reduce_mem_usage
from google.cloud import bigquery_storage_v1beta1
from io import BytesIO


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class BaseFeature(abc.ABC):
    save_memory: bool = True

    def __init__(self, debugging: bool = False, **kwargs) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.debugging = debugging
        self._logger = Logger(self.__class__.__name__)
        handler = StreamHandler()
        fmt = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        handler.setLevel(INFO)
        self._logger.addHandler(handler)

        self.TESTING = TESTING
        self.GCS_BUCKET_NAME = GCS_BUCKET_NAME
        self.PROJECT_ID = PROJECT_ID
        self.train_table = None
        self.test_table = None
        self.train_text = None
        self.test_text = None

    @abc.abstractmethod
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

    @classmethod
    def add_feature_specific_arguments(cls, parser: argparse.ArgumentParser):
        return

    @classmethod
    def main(cls):
        import logging

        logging.basicConfig(level=logging.INFO)
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        cls.add_feature_specific_arguments(parser)
        args = parser.parse_args()
        instance = cls(debugging=args.debug, **vars(args))
        instance.run()

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        self._logger.info(f"Running with debugging={self.debugging}")
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            if TESTING:
                test_path = os.path.join(tempdir, f"{self.name}_test.ftr")
                self.test_table = f"`{PROJECT_ID}.recsys2020.test`"
                self.test_text = f"`{PROJECT_ID}.recsys2020.texts_test`"
            else:
                test_path = os.path.join(tempdir, f"{self.name}_val_20200418.ftr")
                self.test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
                self.test_text = f"`{PROJECT_ID}.recsys2020.texts_val_20200418`"

            train_path = os.path.join(tempdir, f"{self.name}_training.ftr")
            self.train_table = f"`{PROJECT_ID}.recsys2020.training`"
            self.train_text = f"`{PROJECT_ID}.recsys2020.texts_training`"

            self.read_and_save_features(
                self.train_table, self.test_table, train_path, test_path,
            )
            self._upload_to_gs([test_path, train_path])

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
        assert (
            df_test_input.shape[0] == df_test_features.shape[0]
        ), "generated test features is not compatible with the table"
        df_train_features.columns = f"{self.name}_" + df_train_features.columns
        df_test_features.columns = f"{self.name}_" + df_test_features.columns

        if self.save_memory:
            self._logger.info("Reduce memory size - train data")
            df_train_features = reduce_mem_usage(df_train_features)
            self._logger.info("Reduce memory size - test data")
            df_test_features = reduce_mem_usage(df_test_features)

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)
        self._logger.info(f"Saving features to {test_output_path}")
        df_test_features.to_feather(test_output_path)

    def _read_from_bigquery(self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = """
            select {}
            from {}
            order by tweet_id, engaging_user_id
        """.format(
            ", ".join(self.import_columns()), table_name
        )
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def _upload_to_gs(self, files: List[str]):
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(GCS_BUCKET_NAME)

        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        for filename in files:
            basename = os.path.basename(filename)
            blob = storage.Blob(os.path.join(bucket_dir_name, basename), bucket)
            self._logger.info(f"Uploading {basename} to {blob.path}")
            blob.upload_from_filename(filename)

    def _download_from_gs(self, feather_file_name: str) -> pd.DataFrame:
        """GCSにある特徴量ファイル(feather形式)を読み込む
        """
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(GCS_BUCKET_NAME)

        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        blob = storage.Blob(
            os.path.join(bucket_dir_name, feather_file_name),
            bucket
        )
        content = blob.download_as_string()
        print(f"Downloading {feather_file_name} from {blob.path}")
        df = pd.read_feather(BytesIO(content))

        return df
