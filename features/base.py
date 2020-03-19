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
    def __init__(self, name: Optional[str]) -> None:
        super().__init__
        self.name = name or self.__class__.__name__
        self._logger = Logger(self.__class__.__name__)

    @abc.abstractproperty
    def import_columns(self) -> List[str]:
        """この特徴量を作るのに必要なカラムを指定する
        """
        ...

    @abc.abstractmethod
    def make_features(self, df_train_input: pd.DataFrame, df_val_input: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """BigQuery から取得した生データの DataFrame を特徴量に変換する
        """
        ...

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            if TESTING:
                test_path = self.read_and_generate_features("`wantedly-individual-naomichi.recsys2020.test`", os.path.join(tempdir.name, f"{self.name}_test.ftr"))
                files.append(test_path)
            else:
                val_path = self.read_and_generate_features("`wantedly-individual-naomichi.recsys2020.val`", os.path.join(tempdir.name, f"{self.name}_val.ftr"))
                files.append(val_path)
            train_path = self.read_and_generate_features("`wantedly-individual-naomichi.recsys2020.training`", os.path.join(tempdir.name, f"{self.name}_training.ftr"))
            files.append(train_path)
            self._upload_to_gs(files)

    def read_and_generate_features(self, table_name: str, output_path: str) -> str:
        self._logger.info(f"Processing {table_name}")
        query = """
            select {}
            from {}
            order by tweet_id, engaging_user_id
        """.format(self.import_columns.join(", "), table_name)
        self._logger.info("Executing query {}".format(repr(query)))
        df_input = pd.read_gbq(query, dialect="standard", project_id="wantedly-individual-naomichi")
        df_features = self.make_features(df_input)
        assert df_input.shape[0] == df_features.shape[0], "generated features is not compatible with the table"
        self._logger.info(f"Saving features into {output_path}")
        df_features.columns = f"{self.name}_" + df_features.columns
        df_features.to_feather(output_path)
        return output_path

    def _upload_to_gs(self, files: List[str]):
        client = storage.Client(project="wantedly-individual-naomichi")
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        for filename in files:
            basename = os.path.basename(filename)
            blob = bucket.get_blob(os.path.join("features", basename))
            self._logger.info(f"Uploading {basename} to {blob.path}")
            blob.upload_from_filename(filename)
